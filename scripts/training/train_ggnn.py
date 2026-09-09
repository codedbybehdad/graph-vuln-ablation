# ============================================================
# TRAIN_GGNN.PY
# Relation-aware GGNN for vulnerability detection
# ============================================================

import os
import json
import random
import argparse
from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, WeightedRandomSampler
from torch_geometric.loader import DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ------------------------------------------------------------
# Reproducibility / CUDA
# ------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism is important for thesis experiments.  We disable
    # benchmark selection because graph batches have variable shapes.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class GraphDataset(Dataset):
    def __init__(self, items, edge_mode="all"):
        self.items = items
        self.edge_mode = edge_mode.lower()

    def __len__(self):
        return len(self.items)

    def filter_edges(self, data):
        if self.edge_mode == "all":
            return data

        edge_type_map = {"ast": 0, "cfg": 1, "pdg": 2}
        selected = {
            edge_type_map[token.strip()]
            for token in self.edge_mode.split("+")
            if token.strip() in edge_type_map
        }

        if not selected:
            raise ValueError(f"Unknown edge mode: {self.edge_mode}")

        mask = torch.zeros_like(data.edge_type, dtype=torch.bool)
        for edge_id in selected:
            mask |= data.edge_type == edge_id

        # Preserve the complete node set and only remove connections.
        data.edge_index = data.edge_index[:, mask]
        data.edge_type = data.edge_type[mask]
        return data

    def __getitem__(self, idx):
        item = self.items[idx]
        data = torch.load(item["path"], weights_only=False)
        return self.filter_edges(data)


# ------------------------------------------------------------
# Relation-aware GGNN
# ------------------------------------------------------------

class RelationGGNNLayer(nn.Module):
    """One GGNN propagation step with one learned transform per relation."""

    def __init__(self, hidden_dim, num_relations=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.relation_linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_relations)]
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, h, edge_index, edge_type):
        src, dst = edge_index
        aggregated = torch.zeros_like(h)

        for relation_id, linear in enumerate(self.relation_linears):
            mask = edge_type == relation_id
            if not torch.any(mask):
                continue
            relation_src = src[mask]
            relation_dst = dst[mask]
            messages = linear(h[relation_src])
            aggregated.index_add_(0, relation_dst, messages)

        # Paper-style gated recurrent update: new state = GRU(old state, aggregate).
        return self.gru(aggregated, h)


class GGNN(nn.Module):
    """Multi-relational GGNN + Devign-inspired dual Conv1d readout."""

    def __init__(self, in_channels, hidden_dim=200, num_steps=6, num_relations=3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps

        # Devign initializes the hidden state by copying the annotation and
        # padding with zeros when hidden_dim > input_dim.
        self.input_projection = (
            nn.Identity()
            if in_channels == hidden_dim
            else nn.Linear(in_channels, hidden_dim)
        )

        self.ggnn = RelationGGNNLayer(hidden_dim, num_relations=num_relations)

        # The Conv module follows the paper's idea of applying the same 1-D
        # convolution stack to [H^T, X] and H^T and combining both signals.
        concat_channels = hidden_dim + in_channels
        self.conv_z1 = nn.Conv1d(concat_channels, 64, kernel_size=3, padding=1)
        self.conv_z2 = nn.Conv1d(64, 32, kernel_size=1, padding=1)
        self.conv_y1 = nn.Conv1d(hidden_dim, 64, kernel_size=3, padding=1)
        self.conv_y2 = nn.Conv1d(64, 32, kernel_size=1, padding=1)

        self.fc_z = nn.Linear(32, 1)
        self.fc_y = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)

    def _conv_branch(self, seq, conv1, conv2):
        # seq: [nodes, channels]. Treat nodes as the temporal/ordered axis.
        seq = seq.transpose(0, 1).unsqueeze(0)  # [1, C, N]
        seq = F.relu(conv1(seq))
        seq = F.max_pool1d(seq, kernel_size=3, stride=2, ceil_mode=True)
        seq = F.relu(conv2(seq))
        seq = F.max_pool1d(seq, kernel_size=2, stride=2, ceil_mode=True)
        seq = seq.mean(dim=-1).squeeze(0)  # [C]
        return seq

    def forward(self, x, edge_index, edge_type, batch):
        h = self.input_projection(x)

        # Sequential gated message passing.  Edge type is used at every step.
        for _ in range(self.num_steps):
            h = self.ggnn(h, edge_index, edge_type)

        graph_logits = []
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0

        for graph_id in range(num_graphs):
            mask = batch == graph_id
            hg = h[mask]
            xg = x[mask]

            # Devign-style dual branch on the graph's ordered node sequence.
            z_input = torch.cat([hg, xg], dim=1)
            z = self._conv_branch(z_input, self.conv_z1, self.conv_z2)
            y = self._conv_branch(hg, self.conv_y1, self.conv_y2)

            z_logit = self.fc_z(z)
            y_logit = self.fc_y(y)
            pairwise_logit = z_logit * y_logit

            graph_logits.append(pairwise_logit.reshape(1))

        return torch.cat(graph_logits, dim=0)


# ------------------------------------------------------------
# Metrics / evaluation
# ------------------------------------------------------------

def best_threshold_for_f1(probs, labels):
    best_threshold = 0.5
    best_f1 = -1.0

    # Dense but deterministic threshold search.
    for t in np.arange(0.10, 0.91, 0.01):
        preds = (probs >= t).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(t)

    return best_threshold


def evaluate(model, loader, device, threshold=None, select_threshold=False):
    model.eval()

    probs_all = []
    labels_all = []
    logits_all = []

    with torch.inference_mode():
        for data in loader:
            data = data.to(device, non_blocking=True)
            logits = model(data.x, data.edge_index, data.edge_type, data.batch)
            probs = torch.sigmoid(logits)

            probs_all.extend(probs.detach().cpu().numpy())
            labels_all.extend(data.y.detach().cpu().numpy())
            logits_all.extend(logits.detach().cpu().numpy())

    probs_all = np.asarray(probs_all, dtype=np.float64)
    labels_all = np.asarray(labels_all, dtype=np.int64)

    if threshold is None:
        threshold = 0.5

    if select_threshold:
        threshold = best_threshold_for_f1(probs_all, labels_all)

    preds = (probs_all >= threshold).astype(int)

    auc = 0.5 if len(np.unique(labels_all)) < 2 else roc_auc_score(labels_all, probs_all)

    return {
        "accuracy": accuracy_score(labels_all, preds),
        "precision": precision_score(labels_all, preds, zero_division=0),
        "recall": recall_score(labels_all, preds, zero_division=0),
        "f1": f1_score(labels_all, preds, zero_division=0),
        "auc": auc,
        "threshold": float(threshold),
        "logit_mean": float(np.mean(logits_all)),
        "prob_mean": float(np.mean(probs_all)),
        "prob_std": float(np.std(probs_all)),
    }


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, scaler, amp_enabled, grad_clip):
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(data.x, data.edge_index, data.edge_type, data.batch)
            labels = data.y.float().view(-1)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.detach().cpu())

    return total_loss / max(len(loader), 1)


def build_sampler(items):
    labels = np.asarray([item["label"] for item in items], dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2)
    class_weights = np.zeros(2, dtype=np.float64)
    for cls in range(2):
        class_weights[cls] = 1.0 / max(class_counts[cls], 1)
    sample_weights = torch.as_tensor(class_weights[labels], dtype=torch.double)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def make_loader(dataset, batch_size, train, workers, pin_memory, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=build_sampler(dataset.items) if train else None,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker if workers > 0 else None,
        generator=generator,
    )


def train_fold(args, train_items, val_items, fold_id, device):
    fold_seed = args.seed + fold_id
    set_seed(fold_seed)

    train_dataset = GraphDataset(train_items, edge_mode=args.edges)
    val_dataset = GraphDataset(val_items, edge_mode=args.edges)

    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.workers, pin_memory, fold_seed
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, args.workers, pin_memory, fold_seed
    )

    sample = train_dataset[0]
    in_channels = int(sample.x.shape[1])

    model = GGNN(
        in_channels=in_channels,
        hidden_dim=args.hidden_dim,
        num_steps=args.steps,
        num_relations=3,
    ).to(device)

    # The weighted sampler already balances classes, so avoid double
    # reweighting the positive class in BCE.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience
    )

    amp_enabled = bool(device.type == "cuda" and not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if device.type == "cuda" else None

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    best = None
    history = []
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            amp_enabled,
            args.grad_clip,
        )

        val_metrics = evaluate(
            model, val_loader, device, select_threshold=True
        )
        scheduler.step(val_metrics["f1"])

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **val_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        print(
            f"Fold {fold_id} | Epoch {epoch + 1}/{args.epochs} | "
            f"loss={train_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"thr={val_metrics['threshold']:.2f}"
        )

        monitor = (val_metrics["f1"], val_metrics["auc"], val_metrics["accuracy"])
        if best is None or monitor > best["monitor"]:
            best = {
                "monitor": monitor,
                "epoch": epoch + 1,
                "metrics": deepcopy(val_metrics),
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Fold {fold_id} | early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best["state_dict"])
    final_metrics = evaluate(
        model,
        val_loader,
        device,
        threshold=best["metrics"]["threshold"],
        select_threshold=False,
    )

    model_path = os.path.join(
        args.model_dir, f"best_model_{args.dataset}_{args.edges}_fold{fold_id}.pt"
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_channels": in_channels,
            "hidden_dim": args.hidden_dim,
            "steps": args.steps,
            "edges": args.edges,
            "fold": fold_id,
            "seed": fold_seed,
            "threshold": best["metrics"]["threshold"],
        },
        model_path,
    )

    return {
        "fold": fold_id,
        "seed": fold_seed,
        "best_epoch": best["epoch"],
        "metrics": final_metrics,
        "model_path": model_path,
        "history": history,
    }


def aggregate_fold_metrics(fold_results):
    metric_names = ["accuracy", "precision", "recall", "f1", "auc"]
    aggregate = {}
    for name in metric_names:
        values = np.asarray([r["metrics"][name] for r in fold_results], dtype=np.float64)
        aggregate[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=1) if len(values) > 1 else 0.0)}
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["qemu", "ffmpeg"])
    parser.add_argument("--edges", type=str, default="all", choices=[
        "all", "ast", "cfg", "pdg", "ast+cfg", "ast+pdg", "cfg+pdg", "ast+cfg+pdg"
    ])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1.3e-6)
    parser.add_argument("--lr_patience", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n🚀 Device:", device)
    if device.type == "cuda":
        print("🎮 GPU:", torch.cuda.get_device_name(0))
        print("💾 VRAM (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
        print("⚡ AMP:", not args.no_amp)

    dataset_index_path = f"data/processed/{args.dataset}_dataset_index.pt"
    if not os.path.exists(dataset_index_path):
        raise FileNotFoundError(f"Dataset index not found: {dataset_index_path}")

    meta = torch.load(dataset_index_path, weights_only=False)
    dataset_items = meta["graphs"]
    labels = np.asarray([item["label"] for item in dataset_items], dtype=np.int64)

    print(
        f"📦 {args.dataset.upper()} graphs: {len(dataset_items)} | "
        f"positive={int(labels.sum())} negative={int((labels == 0).sum())}"
    )
    print(
        f"🔁 {args.folds}-fold stratified cross-validation | "
        f"edges={args.edges} | batch={args.batch_size} | hidden={args.hidden_dim} | steps={args.steps}"
    )

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    all_fold_results = []

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_items = [dataset_items[i] for i in train_idx]
        val_items = [dataset_items[i] for i in val_idx]
        print("\n" + "=" * 72)
        print(f"FOLD {fold_id}/{args.folds}: train={len(train_items)} val={len(val_items)}")
        print("=" * 72)

        result = train_fold(args, train_items, val_items, fold_id, device)
        all_fold_results.append(result)
        if device.type == "cuda":
            peak_gib = torch.cuda.max_memory_allocated(device) / (2 ** 30)
            print(f"Fold {fold_id} peak VRAM allocated: {peak_gib:.2f} GiB")
        print(f"Fold {fold_id} result: {json.dumps(result['metrics'], indent=2)}")

    aggregate = aggregate_fold_metrics(all_fold_results)

    output = {
        "dataset": args.dataset,
        "edges": args.edges,
        "folds": args.folds,
        "seed": args.seed,
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "steps": args.steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lr_patience": args.lr_patience,
            "patience": args.patience,
            "grad_clip": args.grad_clip,
            "amp": not args.no_amp,
            "workers": args.workers,
        },
        "folds_results": all_fold_results,
        "aggregate": aggregate,
    }

    safe_edges = args.edges.replace("+", "_")
    result_path = os.path.join(
        args.results_dir,
        f"cv5_{args.dataset}_{safe_edges}.json",
    )
    os.makedirs(args.results_dir, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 72)
    print("5-FOLD SUMMARY")
    print("=" * 72)
    for name, stats in aggregate.items():
        print(f"{name.upper():9s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print("\n💾 Results:", result_path)


if __name__ == "__main__":
    main()
