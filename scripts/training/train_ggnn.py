# ============================================================
# TRAIN_GGNN.PY
# Stable GGNN training for vulnerability detection.
#
# Edge modes:
#   all, ast, cfg, pdg, ast+cfg, ast+pdg, cfg+pdg, ast+cfg+pdg
#
# Notes:
# - AST/CFG/PDG are selected by filtering the preprocessed edge_type tensor.
# - GatedGraphConv does not consume edge_type directly; edge_type therefore
#   controls the ablation by deciding which edges reach the GGNN.
# - "all" means use AST+CFG+PDG in one model run.
# - Class balancing is handled by BCEWithLogitsLoss only. A WeightedRandomSampler
#   is deliberately not combined with pos_weight, avoiding double reweighting.
# - Validation threshold is selected by F1, which is less prone than accuracy
#   to rewarding an all-positive prediction on a mildly imbalanced dataset.
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
from torch_geometric.nn import GatedGraphConv, AttentionalAggregation


EDGE_TYPE_MAP = {
    "ast": 0,
    "cfg": 1,
    "pdg": 2,
}
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


# ============================================================
# REPRODUCIBILITY
# ============================================================


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================


class GraphDataset(Dataset):
    def __init__(self, items, edge_mode="all"):
        if edge_mode not in VALID_EDGE_MODES:
            raise ValueError(
                f"Invalid edge mode '{edge_mode}'. "
                f"Expected one of: {', '.join(VALID_EDGE_MODES)}"
            )

        self.items = items
        self.edge_mode = edge_mode

    def __len__(self):
        return len(self.items)

    def _selected_edge_types(self):
        if self.edge_mode == "all":
            return {0, 1, 2}

        return {
            EDGE_TYPE_MAP[token]
            for token in self.edge_mode.split("+")
        }

    def filter_edges(self, data):
        selected_types = self._selected_edge_types()

        if not hasattr(data, "edge_type"):
            raise RuntimeError(
                "Processed graph is missing 'edge_type'. "
                "Rebuild the dataset before training."
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

        # "all" is intentionally explicit: all three edge families are kept.
        # For an ablation, keep only the requested edge families.
        mask = torch.zeros(
            data.edge_type.shape,
            dtype=torch.bool,
            device=data.edge_type.device,
        )
        for edge_type_id in selected_types:
            mask |= data.edge_type == edge_type_id

        data.edge_index = data.edge_index[:, mask]
        data.edge_type = data.edge_type[mask]

        return data

    def __getitem__(self, idx):
        item = self.items[idx]

        data = torch.load(
            item["path"],
            weights_only=False,
        )

        data = self.filter_edges(data)
        return data


# ============================================================
# SPLIT
# ============================================================


def split_dataset(index_items):
    labels = [int(item["label"]) for item in index_items]

    train_idx, val_idx = train_test_split(
        np.arange(len(index_items)),
        test_size=0.25,
        stratify=labels,
        random_state=42,
    )

    train_items = [index_items[int(i)] for i in train_idx]
    val_items = [index_items[int(i)] for i in val_idx]

    return train_items, val_items


# ============================================================
# MODEL
# ============================================================


class GGNN(nn.Module):
    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )

        self.ggnn1 = GatedGraphConv(
            out_channels=hidden_dim,
            num_layers=2,
        )

        self.ggnn2 = GatedGraphConv(
            out_channels=hidden_dim,
            num_layers=2,
        )

        self.att_gate = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.att_pool = AttentionalAggregation(self.att_gate)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, x, edge_index, edge_type, batch):
        # edge_type is intentionally not passed into GatedGraphConv because
        # this PyG layer is untyped; GraphDataset has already filtered the
        # graph according to the requested ablation.
        del edge_type

        x = self.node_encoder(x)

        residual = x
        x = self.ggnn1(x, edge_index)
        x = F.relu(x + residual)
        x = F.dropout(x, p=0.15, training=self.training)

        residual = x
        x = self.ggnn2(x, edge_index)
        x = F.relu(x + residual)

        graph_repr = self.att_pool(x, batch)
        logits = self.classifier(graph_repr)

        return logits.view(-1)


# ============================================================
# METRICS / EVALUATION
# ============================================================


def compute_pos_weight(items):
    labels = np.asarray([int(item["label"]) for item in items], dtype=np.int64)

    positives = int(labels.sum())
    negatives = int(len(labels) - positives)

    if positives == 0 or negatives == 0:
        raise RuntimeError(
            "Training split contains only one class; cannot train a binary classifier."
        )

    return torch.tensor(
        [negatives / positives],
        dtype=torch.float32,
    )


def choose_threshold(labels, probabilities):
    """Choose the validation threshold that maximizes F1."""
    best_f1 = -1.0
    best_threshold = 0.5

    # Dense enough grid for reproducible threshold selection without fitting
    # a threshold model or using any information outside the validation split.
    for threshold in np.arange(0.10, 0.91, 0.01):
        predictions = (probabilities >= threshold).astype(np.int64)
        current_f1 = f1_score(
            labels,
            predictions,
            zero_division=0,
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)

    return best_threshold


def evaluate(model, loader, device, threshold=0.5, select_threshold=False):
    model.eval()

    all_probabilities = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            logits = model(
                data.x,
                data.edge_index,
                data.edge_type,
                data.batch,
            )

            probabilities = torch.sigmoid(logits)

            all_probabilities.extend(
                probabilities.detach().cpu().numpy().tolist()
            )
            all_labels.extend(
                data.y.detach().cpu().numpy().astype(np.int64).tolist()
            )
            all_logits.extend(
                logits.detach().cpu().numpy().tolist()
            )

    probabilities = np.asarray(all_probabilities, dtype=np.float64)
    labels = np.asarray(all_labels, dtype=np.int64)
    logits = np.asarray(all_logits, dtype=np.float64)

    if probabilities.size == 0:
        raise RuntimeError("Validation loader returned no graphs.")

    if select_threshold:
        threshold = choose_threshold(labels, probabilities)

    predictions = (probabilities >= threshold).astype(np.int64)

    if np.unique(labels).size >= 2:
        auc = float(roc_auc_score(labels, probabilities))
    else:
        auc = 0.5

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": auc,
        "threshold": float(threshold),
        "logit_mean": float(logits.mean()),
        "prob_mean": float(probabilities.mean()),
        "prob_std": float(probabilities.std()),
        "positive_prediction_rate": float(predictions.mean()),
    }


# ============================================================
# TRAINING
# ============================================================


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    batches = 0

    for data in loader:
        data = data.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            data.x,
            data.edge_index,
            data.edge_type,
            data.batch,
        )

        labels = data.y.float().view(-1)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss encountered: {loss}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        batches += 1

    if batches == 0:
        raise RuntimeError("Training loader returned no batches.")

    return total_loss / batches


def report_edge_counts(items, edge_mode):
    selected_ids = (
        {0, 1, 2}
        if edge_mode == "all"
        else {EDGE_TYPE_MAP[token] for token in edge_mode.split("+")}
    )

    counts = {0: 0, 1: 0, 2: 0}
    graphs_with_edges = 0

    for item in items:
        data = torch.load(item["path"], weights_only=False)
        edge_type = data.edge_type

        present_in_graph = False
        for edge_type_id in selected_ids:
            count = int((edge_type == edge_type_id).sum().item())
            counts[edge_type_id] += count
            present_in_graph = present_in_graph or count > 0

        if present_in_graph:
            graphs_with_edges += 1

    missing = [
        name
        for name, edge_type_id in EDGE_TYPE_MAP.items()
        if edge_type_id in selected_ids and counts[edge_type_id] == 0
    ]

    if missing:
        raise RuntimeError(
            f"No {', '.join(missing).upper()} edges are available for edge mode '{edge_mode}'."
        )

    print("\n📌 Selected edge types:", edge_mode)
    print(f"   AST edges: {counts[0]}")
    print(f"   CFG edges: {counts[1]}")
    print(f"   PDG edges: {counts[2]}")
    print(f"   Graphs with at least one selected edge: {graphs_with_edges}/{len(items)}")


def run_training(args):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n🚀 Device:", device)
    print("🧩 Edge mode:", args.edges)

    dataset_index_path = os.path.join(
        "data",
        "processed",
        f"{args.dataset}_dataset_index.pt",
    )

    if not os.path.exists(dataset_index_path):
        raise FileNotFoundError(
            f"Dataset index not found: {dataset_index_path}"
        )

    meta = torch.load(dataset_index_path, weights_only=False)
    dataset_items = meta.get("graphs", [])

    if not dataset_items:
        raise RuntimeError("Dataset index contains no graphs.")

    train_items, val_items = split_dataset(dataset_items)

    print(
        f"📦 Graphs: {len(dataset_items)} total | "
        f"{len(train_items)} train | {len(val_items)} validation"
    )

    print("📌 Training label counts:")
    train_labels = [int(item["label"]) for item in train_items]
    print(
        f"   class 0: {train_labels.count(0)} | "
        f"class 1: {train_labels.count(1)}"
    )

    report_edge_counts(train_items, args.edges)

    train_dataset = GraphDataset(train_items, edge_mode=args.edges)
    val_dataset = GraphDataset(val_items, edge_mode=args.edges)

    # Use class-weighted BCE only. Do not also resample the training set.
    pos_weight = compute_pos_weight(train_items).to(device)
    print(f"⚖️ Positive-class weight: {pos_weight.item():.4f}")

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
    in_channels = int(sample.x.shape[1])

    if sample.x.ndim != 2:
        raise RuntimeError(
            f"Expected node features with shape [num_nodes, num_features], got {tuple(sample.x.shape)}"
        )

    model = GGNN(in_channels=in_channels).to(device)

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
        patience=4,
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    model_path = os.path.join(
        "models",
        f"best_model_{args.dataset}_{args.edges}.pt",
    )
    result_path = os.path.join(
        "results",
        f"{args.dataset}_{args.edges}_metrics.json",
    )
    plot_path = os.path.join(
        "results",
        f"{args.dataset}_{args.edges}_training_curve.png",
    )

    best_f1 = -1.0
    best_auc = -1.0
    best_accuracy = -1.0
    best_threshold = 0.5
    best_epoch = 0
    epochs_without_improvement = 0

    train_losses = []
    val_f1s = []
    val_aucs = []
    val_accuracies = []

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            threshold=0.5,
            select_threshold=True,
        )

        scheduler.step(val_metrics["auc"])

        train_losses.append(train_loss)
        val_f1s.append(val_metrics["f1"])
        val_aucs.append(val_metrics["auc"])
        val_accuracies.append(val_metrics["accuracy"])

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        print(f"Val AUC: {val_metrics['auc']:.4f}")
        print(f"Threshold: {val_metrics['threshold']:.2f}")
        print(f"Val Positive Rate: {val_metrics['positive_prediction_rate']:.4f}")
        print(f"Prob Mean: {val_metrics['prob_mean']:.4f}")
        print(f"Prob Std: {val_metrics['prob_std']:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")

        # Primary checkpoint metric: validation AUC.
        # F1 and accuracy are tie-breakers so the checkpoint does not reward
        # a pathological all-positive classifier merely because of thresholding.
        is_better = (
            val_metrics["auc"] > best_auc + 1e-8
            or (
                abs(val_metrics["auc"] - best_auc) <= 1e-8
                and val_metrics["f1"] > best_f1
            )
            or (
                abs(val_metrics["auc"] - best_auc) <= 1e-8
                and abs(val_metrics["f1"] - best_f1) <= 1e-8
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
        raise RuntimeError("No model checkpoint was saved during training.")

    print("\n📥 Loading best model...")
    model.load_state_dict(torch.load(model_path, weights_only=True))

    final_metrics = evaluate(
        model,
        val_loader,
        device,
        threshold=best_threshold,
        select_threshold=False,
    )

    final_metrics.update({
        "dataset": args.dataset,
        "edges": args.edges,
        "evaluation_split": "validation",
        "split_ratio": "75% train / 25% validation",
        "selection_metric": "validation AUC (F1, then accuracy tie-breakers)",
        "reported_metrics": [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
        ],
        "selected_checkpoint_epoch": int(best_epoch),
        "selected_checkpoint_accuracy": float(best_accuracy),
        "selected_checkpoint_f1": float(best_f1),
        "selected_checkpoint_auc": float(best_auc),
        "selected_checkpoint_threshold": float(best_threshold),
        "highest_validation_accuracy": float(max(val_accuracies)),
        "highest_validation_accuracy_percent": float(max(val_accuracies) * 100.0),
        "highest_validation_accuracy_epoch": int(np.argmax(val_accuracies) + 1),
    })

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
        "--dataset",
        required=True,
        choices=["qemu", "ffmpeg"],
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
        help=(
            "Edge configuration. 'all' uses AST+CFG+PDG together; "
            "the other values select one ablation configuration."
        ),
    )

    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
