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
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn import RGCNConv, DataParallel as PyGDataParallel
from torch_geometric.utils import to_dense_batch

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)


# ------------------------------------------------------------
# Reproducibility / CUDA
# ------------------------------------------------------------

def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic kernels are useful for exact reruns, but they can be
    # materially slower on Kaggle. Keep them opt-in for high-throughput runs.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class GraphDataset(Dataset):
    def __init__(self, items, edge_mode="all", graph_cache=None):
        self.items = items
        self.edge_mode = edge_mode.lower()
        # Reuse preloaded Data objects across folds/configurations.
        self.graph_cache = graph_cache

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

        mask = torch.zeros(data.edge_type.numel(), dtype=torch.bool)
        for edge_id in selected:
            mask |= data.edge_type.cpu() == edge_id

        # Build a lightweight shallow copy so the cached graph is never mutated.
        filtered = data.clone()
        filtered.edge_index = data.edge_index[:, mask]
        filtered.edge_type = data.edge_type[mask]
        return filtered

    def __getitem__(self, idx):
        item = self.items[idx]
        if self.graph_cache is not None:
            data = self.graph_cache[item["file_id"]]
        else:
            data = torch.load(item["path"], weights_only=False)
        return self.filter_edges(data)


# ------------------------------------------------------------
# Relation-aware GGNN
# ------------------------------------------------------------

class RelationGGNNLayer(nn.Module):
    """One Devign-style multi-relational GGNN propagation step."""

    def __init__(self, hidden_dim, num_relations=3):
        super().__init__()
        self.rgcn = RGCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,
            aggr="sum",
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, h, edge_index, edge_type):
        messages = self.rgcn(h, edge_index, edge_type)
        return self.gru(messages, h)


class GGNN(nn.Module):
    """Devign-style multi-relational GGNN with sequence-level Conv readout."""

    def __init__(self, in_channels, hidden_dim=200, num_steps=6, num_relations=3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps

        # Devign pads the initial node representation to the GGNN hidden size.
        # A learned projection is used only when the experiment adds features.
        self.input_projection = (
            nn.Identity()
            if in_channels == hidden_dim
            else nn.Linear(in_channels, hidden_dim)
        )
        self.ggnn = RelationGGNNLayer(hidden_dim, num_relations=num_relations)

        self.conv1_z = nn.Conv1d(
            hidden_dim + in_channels, 64, kernel_size=3, padding=1
        )
        self.pool1_z = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2_z = nn.Conv1d(64, 32, kernel_size=1, padding=0)
        self.pool2_z = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.mlp_z = nn.Linear(32, 1)

        self.conv1_y = nn.Conv1d(hidden_dim, 64, kernel_size=3, padding=1)
        self.pool1_y = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2_y = nn.Conv1d(64, 32, kernel_size=1, padding=0)
        self.pool2_y = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.mlp_y = nn.Linear(32, 1)

    @staticmethod
    def _branch(dense_seq, valid, conv1, pool1, conv2, pool2, mlp):
        # dense_seq: [B, N, C], valid: [B, N]
        # Keep the sequence dimension because Devign performs the pairwise
        # multiplication between the two learned sequences before graph-level
        # reduction.
        seq = dense_seq.transpose(1, 2)
        seq = F.relu(conv1(seq))

        valid_f = valid.unsqueeze(1).to(dtype=seq.dtype)
        valid_f = F.max_pool1d(valid_f, kernel_size=3, stride=2, padding=1)
        seq = pool1(seq)

        seq = F.relu(conv2(seq))
        valid_f = F.max_pool1d(valid_f, kernel_size=2, stride=2, padding=1)
        seq = pool2(seq)

        node_logits = mlp(seq.transpose(1, 2)).squeeze(-1)
        valid_f = valid_f.squeeze(1)
        return node_logits, valid_f

    def forward_tensors(self, x, edge_index, edge_type, batch):
        h = self.input_projection(x)
        for _ in range(self.num_steps):
            h = self.ggnn(h, edge_index, edge_type)

        h_dense, valid = to_dense_batch(h, batch)
        x_dense, _ = to_dense_batch(x, batch)

        z, valid_z = self._branch(
            torch.cat([h_dense, x_dense], dim=-1),
            valid,
            self.conv1_z, self.pool1_z, self.conv2_z, self.pool2_z, self.mlp_z,
        )
        y, valid_y = self._branch(
            h_dense,
            valid,
            self.conv1_y, self.pool1_y, self.conv2_y, self.pool2_y, self.mlp_y,
        )

        valid_nodes = (valid_z * valid_y).to(dtype=z.dtype)
        pairwise = z * y
        denom = valid_nodes.sum(dim=1).clamp_min(1.0)
        graph_logits = (pairwise * valid_nodes).sum(dim=1) / denom

        return graph_logits

    def forward(self, x, edge_index=None, edge_type=None, batch=None):
        # Returns logits. Apply sigmoid only for evaluation/metric computation.
        # PyG DataParallel calls the wrapped module with a Data/Batch object.
        if hasattr(x, "x") and hasattr(x, "edge_index"):
            data = x
            return self.forward_tensors(data.x, data.edge_index, data.edge_type, data.batch)
        return self.forward_tensors(x, edge_index, edge_type, batch)


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


def evaluate(model, loader, device, threshold=None, select_threshold=False, multi_gpu=False):
    model.eval()

    probs_all = []
    labels_all = []

    with torch.inference_mode():
        for data in loader:
            if multi_gpu:
                labels = torch.cat([d.y.view(-1).long() for d in data], dim=0)
                logits = model(data)
                probs = torch.sigmoid(logits)
                labels_all.extend(labels.cpu().numpy())
                probs_all.extend(probs.detach().cpu().numpy())
            else:
                data = data.to(device, non_blocking=True)
                logits = model(data.x, data.edge_index, data.edge_type, data.batch)
                probs = torch.sigmoid(logits)
                probs_all.extend(probs.detach().cpu().numpy())
                labels_all.extend(data.y.detach().view(-1).cpu().numpy())

    probs_all = np.asarray(probs_all, dtype=np.float64)
    labels_all = np.asarray(labels_all, dtype=np.int64)

    if threshold is None:
        threshold = 0.5
    if select_threshold:
        threshold = best_threshold_for_f1(probs_all, labels_all)

    preds = (probs_all >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels_all, preds, labels=[0, 1]).ravel()

    auc = 0.5 if len(np.unique(labels_all)) < 2 else roc_auc_score(labels_all, probs_all)
    pr_auc = 0.5 if len(np.unique(labels_all)) < 2 else average_precision_score(labels_all, probs_all)
    mcc = matthews_corrcoef(labels_all, preds) if len(np.unique(labels_all)) > 1 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0

    return {
        "accuracy": float(accuracy_score(labels_all, preds)),
        "precision": float(precision_score(labels_all, preds, zero_division=0)),
        "recall": float(recall_score(labels_all, preds, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(labels_all, preds, zero_division=0)),
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "mcc": float(mcc),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "threshold": float(threshold),
        "prob_mean": float(np.mean(probs_all)),
        "prob_std": float(np.std(probs_all)),
    }


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device, scaler, amp_enabled, grad_clip, multi_gpu=False):
    model.train()
    total_loss = 0.0

    for data in loader:
        optimizer.zero_grad(set_to_none=True)

        if multi_gpu:
            labels = torch.cat([d.y.float().view(-1) for d in data], dim=0)
        else:
            data = data.to(device, non_blocking=True)
            labels = data.y.float().view(-1)

        with torch.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            if multi_gpu:
                logits = model(data)
                loss = criterion(logits, labels.to(logits.device))
            else:
                logits = model(data.x, data.edge_index, data.edge_type, data.batch)
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


def make_loader(dataset, batch_size, train, workers, pin_memory, seed, multi_gpu=False):
    generator = torch.Generator()
    generator.manual_seed(seed)

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=build_sampler(dataset.items) if train else None,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker if workers > 0 else None,
        generator=generator,
    )
    if workers > 0:
        kwargs["prefetch_factor"] = 2
        # Kaggle uses Linux; fork lets workers share the read-only RAM cache
        # without serializing every cached graph into each worker.
        kwargs["multiprocessing_context"] = "fork"
    loader_cls = DataListLoader if multi_gpu else DataLoader
    # DataListLoader is required by PyG DataParallel because it splits whole
    # graph objects, then builds a Batch independently on each GPU.
    return loader_cls(**kwargs)


def train_fold(args, train_items, val_items, fold_id, device, graph_cache, multi_gpu=False):
    fold_seed = args.seed + fold_id
    set_seed(fold_seed, args.deterministic)

    train_dataset = GraphDataset(train_items, edge_mode=args.edges, graph_cache=graph_cache)
    val_dataset = GraphDataset(val_items, edge_mode=args.edges, graph_cache=graph_cache)

    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.workers, pin_memory, fold_seed, multi_gpu=multi_gpu
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, args.workers, pin_memory, fold_seed, multi_gpu=multi_gpu
    )

    sample = train_dataset[0]
    in_channels = int(sample.x.shape[1])

    base_model = GGNN(
        in_channels=in_channels,
        hidden_dim=args.hidden_dim,
        num_steps=args.steps,
        num_relations=3,
    ).to(device)

    model = base_model
    if multi_gpu:
        model = PyGDataParallel(base_model, device_ids=list(range(torch.cuda.device_count())))

    # Train on logits so BCE remains safe under CUDA autocast.
    # The weighted sampler balances classes without double-weighting the loss.
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
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None

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
            multi_gpu=multi_gpu,
        )

        val_metrics = evaluate(
            model, val_loader, device, select_threshold=True, multi_gpu=multi_gpu
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
                "state_dict": {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()},
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Fold {fold_id} | early stopping at epoch {epoch + 1}")
                break

    base_model.load_state_dict(best["state_dict"])
    final_metrics = evaluate(
        model,
        val_loader,
        device,
        threshold=best["metrics"]["threshold"],
        select_threshold=False,
        multi_gpu=multi_gpu,
    )

    model_path = os.path.join(
        args.model_dir, f"best_model_{args.dataset}_{args.edges}_fold{fold_id}.pt"
    )
    torch.save(
        {
            "model_state_dict": base_model.state_dict(),
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
    metric_names = ["accuracy", "precision", "recall", "specificity", "f1", "auc", "pr_auc", "mcc"]
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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1.3e-6)
    parser.add_argument("--lr_patience", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    set_seed(args.seed, args.deterministic)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu = bool(device.type == "cuda" and torch.cuda.device_count() > 1)
    print("\n🚀 Device:", device)
    if device.type == "cuda":
        print("🎮 GPUs:", torch.cuda.device_count())
        for gpu_id in range(torch.cuda.device_count()):
            print(f"   GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)} | VRAM (GiB): {torch.cuda.get_device_properties(gpu_id).total_memory / 2**30:.2f}")
        print("⚡ AMP:", not args.no_amp)
        print("⚡ Multi-GPU:", multi_gpu)

    dataset_index_path = f"data/processed/{args.dataset}_dataset_index.pt"
    if not os.path.exists(dataset_index_path):
        raise FileNotFoundError(f"Dataset index not found: {dataset_index_path}")

    meta = torch.load(dataset_index_path, weights_only=False)
    dataset_items = meta["graphs"]
    labels = np.asarray([item["label"] for item in dataset_items], dtype=np.int64)

    print("💾 Preloading graph objects into RAM for fast CV training...")
    graph_cache = {}
    for i, item in enumerate(dataset_items):
        graph_cache[item["file_id"]] = torch.load(item["path"], weights_only=False)
        if (i + 1) % 1000 == 0:
            print(f"   cached {i + 1}/{len(dataset_items)} graphs")
    print(f"✅ Cached {len(graph_cache)} graphs")

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

        result = train_fold(args, train_items, val_items, fold_id, device, graph_cache, multi_gpu=multi_gpu)
        all_fold_results.append(result)
        if device.type == "cuda":
            peak_gib = [torch.cuda.max_memory_allocated(gpu_id) / (2 ** 30) for gpu_id in range(torch.cuda.device_count())]
            print("Fold {} peak VRAM allocated: {}".format(
                fold_id, ", ".join(f"GPU {i}={v:.2f} GiB" for i, v in enumerate(peak_gib))
            ))
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
            "deterministic": args.deterministic,
            "gpu_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        },
        "folds_results": all_fold_results,
        "aggregate": aggregate,
    }

    safe_edges = args.edges.replace("+", "_")
    result_path = os.path.join(
        args.results_dir,
        f"cv{args.folds}_{args.dataset}_{safe_edges}.json",
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
