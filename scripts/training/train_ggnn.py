# TRAIN_GGNN.PY
# Stable GGNN training for vulnerability detection.
#
# Edge modes:
#   all, ast, cfg, pdg, ast+cfg, ast+pdg, cfg+pdg, ast+cfg+pdg
#
# "all" means AST+CFG+PDG together in one model run.
# ============================================================

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GatedGraphConv, global_max_pool, global_mean_pool


EDGE_TYPE_MAP = {"ast": 0, "cfg": 1, "pdg": 2}
VALID_EDGE_MODES = (
    "all",
    "ast",
    "cfg",
    "pdg",
    "ast+cfg",
    "ast+pdg",
    "cfg+pdg",
    "ast+cfg+pdg",
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GraphDataset(Dataset):
    def __init__(self, items, edge_mode="all"):
        if edge_mode not in VALID_EDGE_MODES:
            raise ValueError(
                f"Invalid edge mode '{edge_mode}'. Expected one of: "
                + ", ".join(VALID_EDGE_MODES)
            )
        self.items = items
        self.edge_mode = edge_mode

    def __len__(self):
        return len(self.items)

    def selected_edge_types(self):
        if self.edge_mode == "all":
            return {0, 1, 2}
        return {EDGE_TYPE_MAP[token] for token in self.edge_mode.split("+")}

    def filter_edges(self, data):
        if not hasattr(data, "edge_type"):
            raise RuntimeError(
                "Processed graph is missing 'edge_type'. Rebuild the dataset."
            )

        if data.edge_index.ndim != 2 or data.edge_index.shape[0] != 2:
            raise RuntimeError(
                f"Invalid edge_index shape: {tuple(data.edge_index.shape)}"
            )

        if data.edge_index.shape[1] != data.edge_type.numel():
            raise RuntimeError(
                "edge_index and edge_type lengths do not match: "
                f"{data.edge_index.shape[1]} vs {data.edge_type.numel()}"
            )

        selected = self.selected_edge_types()
        mask = torch.zeros(data.edge_type.shape, dtype=torch.bool)
        for edge_id in selected:
            mask |= data.edge_type == edge_id

        # Never silently fall back to the original graph for a missing edge type.
        data.edge_index = data.edge_index[:, mask]
        data.edge_type = data.edge_type[mask]
        return data

    def __getitem__(self, idx):
        data = torch.load(self.items[idx]["path"], weights_only=False)
        return self.filter_edges(data)


def split_dataset(index_items):
    labels = np.asarray([int(item["label"]) for item in index_items], dtype=np.int64)
    indices = np.arange(len(index_items))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.25,
        stratify=labels,
        random_state=42,
    )

    return (
        [index_items[int(i)] for i in train_idx],
        [index_items[int(i)] for i in val_idx],
    )


class GGNN(nn.Module):
    """
    GGNN with stable graph-level pooling.

    The previous implementation used attention pooling as the only graph
    representation. That can collapse to nearly identical graph scores when
    the node representations are initially similar. Mean+max pooling preserves
    both the average graph signal and strong local activations and gives the
    classifier a much stronger, stable gradient path.
    """

    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
        )

        self.ggnn1 = GatedGraphConv(
            out_channels=hidden_dim,
            num_layers=2,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.ggnn2 = GatedGraphConv(
            out_channels=hidden_dim,
            num_layers=2,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Mean + max graph pooling = 2 * hidden_dim.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x, edge_index, edge_type, batch):
        # GatedGraphConv in this architecture is untyped; edge_type is already
        # applied by GraphDataset.filter_edges before the message passing layer.
        del edge_type

        x = self.node_encoder(x)

        residual = x
        x = self.ggnn1(x, edge_index)
        x = self.norm1(x + residual)
        x = F.relu(x)

        residual = x
        x = self.ggnn2(x, edge_index)
        x = self.norm2(x + residual)
        x = F.relu(x)

        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        graph_repr = torch.cat([mean_pool, max_pool], dim=1)

        return self.classifier(graph_repr).view(-1)


def evaluate(model, loader, device, threshold=0.5, select_threshold=False):
    model.eval()
    probabilities_all = []
    labels_all = []
    logits_all = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.edge_type, data.batch)
            probabilities = torch.sigmoid(logits)

            probabilities_all.extend(probabilities.cpu().numpy().tolist())
            labels_all.extend(data.y.view(-1).cpu().numpy().astype(np.int64).tolist())
            logits_all.extend(logits.cpu().numpy().tolist())

    probabilities = np.asarray(probabilities_all, dtype=np.float64)
    labels = np.asarray(labels_all, dtype=np.int64)
    logits = np.asarray(logits_all, dtype=np.float64)

    if probabilities.size == 0:
        raise RuntimeError("Validation loader returned no graphs.")
    if np.unique(labels).size < 2:
        raise RuntimeError("Validation split contains only one class.")

    auc = float(roc_auc_score(labels, probabilities))

    if select_threshold:
        # Optimize F1 on validation only. This is used only for reporting,
        # while checkpoint selection uses AUC and therefore does not depend on
        # a potentially pathological threshold.
        best_f1 = -1.0
        best_threshold = 0.5
        for t in np.arange(0.10, 0.91, 0.01):
            preds = (probabilities >= t).astype(np.int64)
            score = f1_score(labels, preds, zero_division=0)
            if score > best_f1:
                best_f1 = float(score)
                best_threshold = float(t)
        threshold = best_threshold

    predictions = (probabilities >= threshold).astype(np.int64)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": auc,
        "threshold": float(threshold),
        "logit_mean": float(logits.mean()),
        "logit_std": float(logits.std()),
        "prob_mean": float(probabilities.mean()),
        "prob_std": float(probabilities.std()),
        "positive_prediction_rate": float(predictions.mean()),
    }


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    batches = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(data.x, data.edge_index, data.edge_type, data.batch)
        labels = data.y.float().view(-1)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {loss.item()}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += float(loss.item())
        batches += 1

    if batches == 0:
        raise RuntimeError("Training loader returned no batches.")
    return total_loss / batches


def compute_pos_weight(items):
    labels = np.asarray([int(item["label"]) for item in items], dtype=np.int64)
    pos = int(labels.sum())
    neg = int(len(labels) - pos)

    if pos == 0 or neg == 0:
        raise RuntimeError(
            f"Training split must contain both classes. Found class 0={neg}, class 1={pos}."
        )

    # Keep the weight bounded to prevent a minority class from dominating the
    # gradients. Do not combine this with a weighted sampler.
    return float(min(neg / pos, 5.0))


def report_dataset(train_items, val_items, edge_mode):
    selected = (
        {0, 1, 2}
        if edge_mode == "all"
        else {EDGE_TYPE_MAP[t] for t in edge_mode.split("+")}
    )

    names = {0: "AST", 1: "CFG", 2: "PDG"}
    edge_counts = {0: 0, 1: 0, 2: 0}
    graph_counts = {0: 0, 1: 0, 2: 0}

    for item in train_items + val_items:
        data = torch.load(item["path"], weights_only=False)
        edge_type = data.edge_type.view(-1)
        for edge_id in selected:
            count = int((edge_type == edge_id).sum().item())
            edge_counts[edge_id] += count
            if count:
                graph_counts[edge_id] += 1

    for edge_id in selected:
        if edge_counts[edge_id] == 0:
            raise RuntimeError(
                f"No {names[edge_id]} edges exist in the processed dataset; "
                f"edge mode '{edge_mode}' cannot be trained."
            )

    print(f"\n📌 Edge mode: {edge_mode}")
    for edge_id in sorted(selected):
        print(
            f"   {names[edge_id]} edges: {edge_counts[edge_id]} "
            f"across {graph_counts[edge_id]} graphs"
        )


def run_training(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n🚀 Device:", device)
    print("🧩 Edge mode:", args.edges)

    dataset_index_path = os.path.join(
        "data", "processed", f"{args.dataset}_dataset_index.pt"
    )
    if not os.path.exists(dataset_index_path):
        raise FileNotFoundError(f"Dataset index not found: {dataset_index_path}")

    meta = torch.load(dataset_index_path, weights_only=False)
    items = meta.get("graphs", [])
    if not items:
        raise RuntimeError("Dataset index contains no graphs.")

    train_items, val_items = split_dataset(items)

    print(
        f"📦 Graphs: {len(items)} total | "
        f"{len(train_items)} train | {len(val_items)} validation"
    )

    train_labels = [int(x["label"]) for x in train_items]
    val_labels = [int(x["label"]) for x in val_items]
    print(
        f"📊 Train labels: class 0={train_labels.count(0)}, "
        f"class 1={train_labels.count(1)}"
    )
    print(
        f"📊 Val labels:   class 0={val_labels.count(0)}, "
        f"class 1={val_labels.count(1)}"
    )

    report_dataset(train_items, val_items, args.edges)

    train_dataset = GraphDataset(train_items, args.edges)
    val_dataset = GraphDataset(val_items, args.edges)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    sample = train_dataset[0]
    if sample.x.ndim != 2:
        raise RuntimeError(
            f"Expected x with shape [nodes, features], got {tuple(sample.x.shape)}"
        )
    in_channels = int(sample.x.shape[1])

    model = GGNN(in_channels=in_channels).to(device)

    # One balancing mechanism only: class-weighted BCE. This avoids the double
    # reweighting that the original sampler + pos_weight combination introduced.
    pos_weight_value = compute_pos_weight(train_items)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    print(f"⚖️ Positive-class weight: {pos_weight_value:.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    model_path = os.path.join(
        "models", f"best_model_{args.dataset}_{args.edges}.pt"
    )
    result_path = os.path.join(
        "results", f"{args.dataset}_{args.edges}_metrics.json"
    )
    plot_path = os.path.join(
        "results", f"{args.dataset}_{args.edges}_training_curve.png"
    )

    best_auc = -float("inf")
    best_f1 = -float("inf")
    best_accuracy = -float("inf")
    best_threshold = 0.5
    best_epoch = 0
    epochs_without_improvement = 0

    train_losses = []
    val_f1s = []
    val_aucs = []
    val_accuracies = []

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            threshold=0.5,
            select_threshold=True,
        )

        # AUC is threshold-independent, so it is the primary checkpoint metric.
        scheduler.step(val_metrics["auc"])

        train_losses.append(train_loss)
        val_f1s.append(val_metrics["f1"])
        val_aucs.append(val_metrics["auc"])
        val_accuracies.append(val_metrics["accuracy"])

        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        print(f"Val AUC: {val_metrics['auc']:.4f}")
        print(f"Threshold: {val_metrics['threshold']:.2f}")
        print(f"Val Positive Rate: {val_metrics['positive_prediction_rate']:.4f}")
        print(f"Logit Mean: {val_metrics['logit_mean']:.4f}")
        print(f"Logit Std: {val_metrics['logit_std']:.4f}")
        print(f"Prob Mean: {val_metrics['prob_mean']:.4f}")
        print(f"Prob Std: {val_metrics['prob_std']:.4f}")
        print(f"Learning Rate: {lr:.6f}")

        is_better = (
            val_metrics["auc"] > best_auc + 1e-6
            or (
                abs(val_metrics["auc"] - best_auc) <= 1e-6
                and val_metrics["f1"] > best_f1 + 1e-6
            )
            or (
                abs(val_metrics["auc"] - best_auc) <= 1e-6
                and abs(val_metrics["f1"] - best_f1) <= 1e-6
                and val_metrics["accuracy"] > best_accuracy
            )
        )

        if is_better:
            best_auc = val_metrics["auc"]
            best_f1 = val_metrics["f1"]
            best_accuracy = val_metrics["accuracy"]
            best_threshold = val_metrics["threshold"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
            print("✅ Best model updated")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\n⏹ Early stopping after {epoch + 1} epochs")
                break

    if best_epoch == 0:
        raise RuntimeError("No model checkpoint was saved.")

    print("\n📥 Loading best model...")
    model.load_state_dict(torch.load(model_path, weights_only=True))

    final_metrics = evaluate(
        model,
        val_loader,
        device,
        threshold=best_threshold,
        select_threshold=False,
    )

    final_metrics.update(
        {
            "dataset": args.dataset,
            "edges": args.edges,
            "evaluation_split": "validation",
            "split_ratio": "75% train / 25% validation",
            "selection_metric": "validation AUC (F1, then accuracy tie-breakers)",
            "selected_checkpoint_epoch": int(best_epoch),
            "selected_checkpoint_accuracy": float(best_accuracy),
            "selected_checkpoint_f1": float(best_f1),
            "selected_checkpoint_auc": float(best_auc),
            "selected_checkpoint_threshold": float(best_threshold),
            "highest_validation_accuracy": float(max(val_accuracies)),
            "highest_validation_accuracy_percent": float(max(val_accuracies) * 100),
            "highest_validation_accuracy_epoch": int(np.argmax(val_accuracies) + 1),
        }
    )

    print("\n✅ FINAL VALIDATION RESULTS\n")
    print(json.dumps(final_metrics, indent=4))

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4)

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_f1s, label="Val F1")
    plt.plot(val_aucs, label="Val AUC")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title(f"{args.dataset.upper()} - {args.edges}")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"\n📁 Results saved to {result_path}")
    print(f"📈 Training curve saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, choices=["qemu", "ffmpeg"]
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument(
        "--edges",
        type=str,
        default="all",
        choices=VALID_EDGE_MODES,
        help="all=AST+CFG+PDG; otherwise choose an individual/composite ablation",
    )
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
