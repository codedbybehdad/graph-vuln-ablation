# train_ggnn.py
# GGNN training for vulnerability detection with stratified K-fold CV.
#
# Edge modes:
#   ast, cfg, pdg, ast+cfg, ast+pdg, cfg+pdg, ast+cfg+pdg
#
# --edges all runs all seven ablation configurations.
#
# Cross-validation:
#   By default the script uses 5-fold stratified cross-validation.
#   The SAME fold assignments are reused for every edge configuration so the
#   ablation study compares models on identical train/validation partitions.
#
# Important:
#   K-fold cross-validation changes how performance is estimated; it does not
#   force AST+CFG+PDG to outperform the other configurations. The code below
#   reports the measured mean and standard deviation across folds.

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
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GatedGraphConv,
    global_max_pool,
    global_mean_pool,
)


EDGE_TYPE_MAP = {"ast": 0, "cfg": 1, "pdg": 2}
EDGE_NAMES = {0: "AST", 1: "CFG", 2: "PDG"}

SINGLE_AND_COMBO_MODES = (
    "ast",
    "cfg",
    "pdg",
    "ast+cfg",
    "ast+pdg",
    "cfg+pdg",
    "ast+cfg+pdg",
)

VALID_EDGE_MODES = ("all",) + SINGLE_AND_COMBO_MODES

CORE_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def expand_edge_modes(edge_mode):
    if edge_mode == "all":
        return list(SINGLE_AND_COMBO_MODES)
    if edge_mode not in SINGLE_AND_COMBO_MODES:
        raise ValueError(
            f"Invalid edge mode '{edge_mode}'. Expected one of: "
            + ", ".join(VALID_EDGE_MODES)
        )
    return [edge_mode]


def edge_ids_for_mode(edge_mode):
    return tuple(EDGE_TYPE_MAP[token] for token in edge_mode.split("+"))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class GraphDataset(Dataset):
    def __init__(self, items, edge_mode):
        if edge_mode not in SINGLE_AND_COMBO_MODES:
            raise ValueError(
                f"Training mode must be a concrete mode, got '{edge_mode}'"
            )
        self.items = items
        self.edge_mode = edge_mode
        self.selected = set(edge_ids_for_mode(edge_mode))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        data = torch.load(self.items[idx]["path"], weights_only=False)

        if not hasattr(data, "edge_type"):
            raise RuntimeError(
                "Processed graph is missing 'edge_type'. Rebuild the dataset."
            )

        if data.edge_index.ndim != 2 or data.edge_index.shape[0] != 2:
            raise RuntimeError(
                f"Invalid edge_index shape: {tuple(data.edge_index.shape)}"
            )

        data.edge_type = data.edge_type.view(-1).long()

        if data.edge_index.shape[1] != data.edge_type.numel():
            raise RuntimeError(
                "edge_index and edge_type lengths do not match: "
                f"{data.edge_index.shape[1]} vs {data.edge_type.numel()}"
            )

        if data.x.ndim != 2 or data.x.numel() == 0:
            raise RuntimeError("Processed graph has invalid/empty node features.")

        if not torch.isfinite(data.x).all():
            raise RuntimeError("Processed graph contains NaN/Inf node features.")

        selected_mask = torch.zeros(
            data.edge_type.shape, dtype=torch.bool
        )
        for edge_id in self.selected:
            selected_mask |= data.edge_type == edge_id

        data.edge_index = data.edge_index[:, selected_mask]
        data.edge_type = data.edge_type[selected_mask]

        return data


def make_stratified_kfold_splits(index_items, n_splits=5, seed=42):
    """
    Create reproducible, stratified K-fold partitions.

    Every graph is used once as validation and n_splits-1 times for training.
    The returned list is independent of edge configuration so the exact same
    partitions are used across all ablation runs.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2 for K-fold cross-validation.")

    if len(index_items) < n_splits:
        raise ValueError(
            f"Cannot create {n_splits} folds from only {len(index_items)} graphs."
        )

    labels = np.asarray(
        [int(item["label"]) for item in index_items],
        dtype=np.int64,
    )

    class_counts = np.bincount(labels, minlength=2)
    if class_counts.min() < n_splits:
        raise ValueError(
            "Each class must contain at least n_splits samples for stratified "
            f"{n_splits}-fold CV. Class counts: "
            f"class 0={class_counts[0]}, class 1={class_counts[1]}"
        )

    indices = np.arange(len(index_items))

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    folds = []

    for fold_number, (train_idx, val_idx) in enumerate(
        splitter.split(indices, labels),
        start=1,
    ):
        train_items = [index_items[int(i)] for i in train_idx]
        val_items = [index_items[int(i)] for i in val_idx]

        train_labels = [int(item["label"]) for item in train_items]
        val_labels = [int(item["label"]) for item in val_items]

        if len(set(train_labels)) < 2 or len(set(val_labels)) < 2:
            raise RuntimeError(
                f"Fold {fold_number} does not contain both classes."
            )

        folds.append(
            {
                "fold": fold_number,
                "train": train_items,
                "val": val_items,
            }
        )

    return folds


def save_fold_assignments(items, folds, dataset, n_splits, seed):
    """Save exact fold membership for reproducibility and auditing."""
    os.makedirs("results", exist_ok=True)

    assignments = {}
    for fold in folds:
        for item in fold["train"]:
            file_id = str(item["file_id"])
            assignments.setdefault(file_id, {})["folds_train"] = assignments.get(
                file_id, {}
            ).get("folds_train", []) + [fold["fold"]]
        for item in fold["val"]:
            file_id = str(item["file_id"])
            assignments.setdefault(file_id, {})["validation_fold"] = fold["fold"]

    output = {
        "dataset": dataset,
        "n_splits": int(n_splits),
        "seed": int(seed),
        "graphs": int(len(items)),
        "fold_assignments": assignments,
    }

    path = os.path.join(
        "results",
        f"{dataset}_{n_splits}fold_split_assignments.json",
    )

    with open(path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print(f"\n🧾 Fold assignments saved to {path}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TypedGGNN(nn.Module):
    """
    One independent GGNN branch for every selected edge type.

    For AST+CFG:
        AST edges -> AST GGNN
        CFG edges -> CFG GGNN
        branch representations -> graph classifier

    The all-three configuration therefore uses three independent typed
    branches, followed by the same classifier design used by the ablations.
    """

    def __init__(self, in_channels, edge_mode, hidden_dim=128, steps=2):
        super().__init__()

        self.edge_mode = edge_mode
        self.edge_ids = edge_ids_for_mode(edge_mode)

        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.branches = nn.ModuleDict()
        self.branch_norms = nn.ModuleDict()

        for edge_id in self.edge_ids:
            key = str(edge_id)
            self.branches[key] = GatedGraphConv(
                out_channels=hidden_dim,
                num_layers=steps,
                aggr="add",
            )
            self.branch_norms[key] = nn.LayerNorm(hidden_dim)

        branch_count = len(self.edge_ids)

        pooled_dim = (
            hidden_dim * (2 + 2 * branch_count)
            + (1 + branch_count)
        )

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, x, edge_index, edge_type, batch):
        x0 = self.node_encoder(x)

        representations = [
            global_mean_pool(x0, batch),
            global_max_pool(x0, batch),
        ]

        node_counts = torch.bincount(
            batch,
            minlength=int(batch.max().item()) + 1,
        )
        structural_features = [
            torch.log1p(node_counts.float()).unsqueeze(1)
        ]

        for edge_id in self.edge_ids:
            mask = edge_type == edge_id
            typed_edges = edge_index[:, mask]

            h = self.branches[str(edge_id)](x0, typed_edges)
            h = F.gelu(self.branch_norms[str(edge_id)](h + x0))

            representations.append(global_mean_pool(h, batch))
            representations.append(global_max_pool(h, batch))

            edge_counts = torch.bincount(
                batch[typed_edges[0]] if typed_edges.numel() else batch[:0],
                minlength=node_counts.numel(),
            ).float()

            structural_features.append(torch.log1p(edge_counts).unsqueeze(1))

        structural = torch.cat(structural_features, dim=1)
        graph_repr = torch.cat(
            representations + [structural],
            dim=1,
        )

        expected_dim = self.classifier[0].in_features
        actual_dim = graph_repr.shape[1]
        if actual_dim != expected_dim:
            raise RuntimeError(
                f"Graph representation dimension mismatch for edge mode "
                f"'{self.edge_mode}': expected {expected_dim}, got {actual_dim}."
            )

        return self.classifier(graph_repr).view(-1)


# ---------------------------------------------------------------------------
# Metrics / training
# ---------------------------------------------------------------------------


def evaluate(model, loader, device, threshold=0.5, select_threshold=False):
    model.eval()

    probabilities_all = []
    labels_all = []
    logits_all = []

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

            probabilities_all.extend(
                probabilities.cpu().numpy().tolist()
            )
            labels_all.extend(
                data.y.view(-1).cpu().numpy().astype(np.int64).tolist()
            )
            logits_all.extend(
                logits.cpu().numpy().tolist()
            )

    probabilities = np.asarray(probabilities_all, dtype=np.float64)
    labels = np.asarray(labels_all, dtype=np.int64)
    logits = np.asarray(logits_all, dtype=np.float64)

    if probabilities.size == 0:
        raise RuntimeError("Validation loader returned no graphs.")

    if np.unique(labels).size < 2:
        raise RuntimeError("Validation split contains only one class.")

    auc = float(roc_auc_score(labels, probabilities))

    if select_threshold:
        # Reporting threshold only. Model/checkpoint selection remains AUC.
        best_f1 = -1.0
        best_threshold = 0.5

        for threshold_candidate in np.arange(0.20, 0.81, 0.01):
            candidate_predictions = (
                probabilities >= threshold_candidate
            ).astype(np.int64)

            candidate_f1 = f1_score(
                labels,
                candidate_predictions,
                zero_division=0,
            )

            if candidate_f1 > best_f1:
                best_f1 = float(candidate_f1)
                best_threshold = float(threshold_candidate)

        threshold = best_threshold

    predictions = (probabilities >= threshold).astype(np.int64)

    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(labels, predictions, zero_division=0)
        ),
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

        logits = model(
            data.x,
            data.edge_index,
            data.edge_type,
            data.batch,
        )

        labels = data.y.float().view(-1)
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss: {loss.item()}"
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )
        optimizer.step()

        total_loss += float(loss.item())
        batches += 1

    if batches == 0:
        raise RuntimeError("Training loader returned no batches.")

    return total_loss / batches


def report_dataset(items, edge_mode):
    selected = edge_ids_for_mode(edge_mode)

    edge_counts = {edge_id: 0 for edge_id in selected}
    graph_counts = {edge_id: 0 for edge_id in selected}

    for item in items:
        data = torch.load(
            item["path"],
            weights_only=False,
        )
        edge_type = data.edge_type.view(-1).long()

        for edge_id in selected:
            count = int(
                (edge_type == edge_id).sum().item()
            )
            edge_counts[edge_id] += count

            if count:
                graph_counts[edge_id] += 1

    print(f"\n📌 Edge mode: {edge_mode}")

    for edge_id in selected:
        if edge_counts[edge_id] == 0:
            raise RuntimeError(
                f"No {EDGE_NAMES[edge_id]} edges exist in the processed "
                f"dataset; mode '{edge_mode}' cannot be trained."
            )

        print(
            f"   {EDGE_NAMES[edge_id]} edges: {edge_counts[edge_id]} "
            f"across {graph_counts[edge_id]} graphs"
        )


def run_single_training(
    args,
    train_items,
    val_items,
    edge_mode,
    device,
    fold_number,
):
    """Train one edge configuration on one CV fold."""
    train_dataset = GraphDataset(train_items, edge_mode)
    val_dataset = GraphDataset(val_items, edge_mode)

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

    model = TypedGGNN(
        in_channels=in_channels,
        edge_mode=edge_mode,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    safe_mode = edge_mode.replace("+", "_")
    run_tag = f"{args.dataset}_{safe_mode}_fold{fold_number}"

    model_path = os.path.join(
        "models",
        f"best_model_{run_tag}.pt",
    )

    result_path = os.path.join(
        "results",
        f"{run_tag}_metrics.json",
    )

    plot_path = os.path.join(
        "results",
        f"{run_tag}_training_curve.png",
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

        lr = optimizer.param_groups[0]["lr"]

        print(f"\nFold {fold_number} | Epoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        print(f"Val AUC: {val_metrics['auc']:.4f}")
        print(f"Threshold: {val_metrics['threshold']:.2f}")
        print(
            f"Val Positive Rate: "
            f"{val_metrics['positive_prediction_rate']:.4f}"
        )
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

            torch.save(
                model.state_dict(),
                model_path,
            )
            print("✅ Best model updated")
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print(
                    f"\n⏹ Early stopping after {epoch + 1} epochs"
                )
                break

    if best_epoch == 0:
        raise RuntimeError(
            f"No model checkpoint was saved for edge mode '{edge_mode}', "
            f"fold {fold_number}."
        )

    print(f"\n📥 Loading best model for fold {fold_number}...")

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    )

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
            "edges": edge_mode,
            "evaluation_split": f"fold_{fold_number}_validation",
            "cross_validation": "StratifiedKFold",
            "n_folds": int(args.folds),
            "fold": int(fold_number),
            "fold_train_size": int(len(train_items)),
            "fold_validation_size": int(len(val_items)),
            "split_ratio": "80% train / 20% validation per fold",
            "selection_metric": (
                "validation AUC "
                "(F1, then accuracy tie-breakers)"
            ),
            "selected_checkpoint_epoch": int(best_epoch),
            "selected_checkpoint_accuracy": float(best_accuracy),
            "selected_checkpoint_f1": float(best_f1),
            "selected_checkpoint_auc": float(best_auc),
            "selected_checkpoint_threshold": float(best_threshold),
            "highest_validation_accuracy": float(
                max(val_accuracies)
            ),
            "highest_validation_accuracy_percent": float(
                max(val_accuracies) * 100
            ),
            "highest_validation_accuracy_epoch": int(
                np.argmax(val_accuracies) + 1
            ),
        }
    )

    print("\n✅ FINAL FOLD VALIDATION RESULTS\n")
    print(json.dumps(final_metrics, indent=4))

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_metrics,
            file,
            indent=4,
        )

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_f1s, label="Val F1")
    plt.plot(val_aucs, label="Val AUC")
    plt.plot(
        val_accuracies,
        label="Val Accuracy",
    )
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title(
        f"{args.dataset.upper()} - {edge_mode} - Fold {fold_number}"
    )
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"\n📁 Fold results saved to {result_path}")
    print(f"📈 Fold training curve saved to {plot_path}")

    return final_metrics


def aggregate_fold_results(fold_results, args, edge_mode):
    """Compute mean/std across the K held-out validation folds."""
    if len(fold_results) != args.folds:
        raise RuntimeError(
            f"Expected {args.folds} fold results for '{edge_mode}', "
            f"received {len(fold_results)}."
        )

    means = {}
    stds = {}

    for metric in CORE_METRICS:
        values = np.asarray(
            [float(result[metric]) for result in fold_results],
            dtype=np.float64,
        )
        means[metric] = float(values.mean())
        stds[metric] = float(values.std(ddof=1))

    threshold_values = np.asarray(
        [float(result["threshold"]) for result in fold_results],
        dtype=np.float64,
    )
    epoch_values = np.asarray(
        [float(result["selected_checkpoint_epoch"]) for result in fold_results],
        dtype=np.float64,
    )

    aggregate = {
        "dataset": args.dataset,
        "edges": edge_mode,
        "cross_validation": "StratifiedKFold",
        "n_folds": int(args.folds),
        "seed": int(args.seed),
        "evaluation_protocol": (
            f"{args.folds}-fold stratified cross-validation; "
            "each fold is used once as validation"
        ),
        "fold_results": fold_results,
        "mean": means,
        "std": stds,
        "threshold_mean": float(threshold_values.mean()),
        "threshold_std": float(threshold_values.std(ddof=1)),
        "selected_epoch_mean": float(epoch_values.mean()),
        "selected_epoch_std": float(epoch_values.std(ddof=1)),
    }

    safe_mode = edge_mode.replace("+", "_")
    path = os.path.join(
        "results",
        f"{args.dataset}_{safe_mode}_{args.folds}fold_metrics.json",
    )

    with open(path, "w", encoding="utf-8") as file:
        json.dump(aggregate, file, indent=4)

    print(f"\n📊 {args.folds}-FOLD SUMMARY: {edge_mode}")
    for metric in CORE_METRICS:
        print(
            f"   {metric.upper():9s}: "
            f"{means[metric]:.4f} ± {stds[metric]:.4f}"
        )
    print(f"   THRESHOLD : {aggregate['threshold_mean']:.3f} ± {aggregate['threshold_std']:.3f}")
    print(f"   EPOCH     : {aggregate['selected_epoch_mean']:.1f} ± {aggregate['selected_epoch_std']:.1f}")
    print(f"\n📁 CV summary saved to {path}")

    return aggregate


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_experiments(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n🚀 Device:", device)
    print(
        f"🔁 Cross-validation: {args.folds}-fold stratified CV "
        f"(seed={args.seed})"
    )

    dataset_index_path = os.path.join(
        "data",
        "processed",
        f"{args.dataset}_dataset_index.pt",
    )

    if not os.path.exists(dataset_index_path):
        raise FileNotFoundError(
            f"Dataset index not found: {dataset_index_path}"
        )

    meta = torch.load(
        dataset_index_path,
        weights_only=False,
    )

    items = meta.get("graphs", [])

    if not items:
        raise RuntimeError(
            "Dataset index contains no graphs."
        )

    modes = expand_edge_modes(args.edges)

    print(
        f"📦 Graphs: {len(items)} total | "
        f"{args.folds} folds | "
        f"~{100 * (args.folds - 1) / args.folds:.1f}% train / "
        f"~{100 / args.folds:.1f}% validation per fold"
    )

    labels = np.asarray(
        [int(item["label"]) for item in items],
        dtype=np.int64,
    )
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique.tolist(), counts.tolist()))
    print(
        "📊 Full dataset labels: "
        + ", ".join(
            f"class {label}={count}" for label, count in label_counts.items()
        )
    )

    # Create folds ONCE and reuse them for every edge configuration.
    folds = make_stratified_kfold_splits(
        items,
        n_splits=args.folds,
        seed=args.seed,
    )

    save_fold_assignments(
        items,
        folds,
        args.dataset,
        args.folds,
        args.seed,
    )

    print(
        "\n🧪 Configurations to run:",
        ", ".join(modes),
    )

    results = {}

    # Report edge availability once over the full dataset.
    for edge_mode in modes:
        report_dataset(items, edge_mode)

    # IMPORTANT: each edge mode gets the same fold partitions.
    for run_number, edge_mode in enumerate(modes, start=1):
        print(
            "\n"
            + "=" * 72
            + f"\nRUN {run_number}/{len(modes)}: {edge_mode}"
            + "\n"
            + "=" * 72
        )

        fold_results = []

        for fold in folds:
            fold_number = fold["fold"]

            print(
                "\n"
                + "-" * 72
                + f"\n{edge_mode.upper()} | FOLD {fold_number}/{args.folds}"
                + "\n"
                + "-" * 72
            )

            train_items = fold["train"]
            val_items = fold["val"]

            train_labels = [int(x["label"]) for x in train_items]
            val_labels = [int(x["label"]) for x in val_items]

            print(
                f"📚 Train: {len(train_items)} graphs | "
                f"class 0={train_labels.count(0)}, "
                f"class 1={train_labels.count(1)}"
            )
            print(
                f"🧪 Val:   {len(val_items)} graphs | "
                f"class 0={val_labels.count(0)}, "
                f"class 1={val_labels.count(1)}"
            )

            # Use the same fold-specific seed for every edge mode so that the
            # random initialization/shuffling is comparable within a fold.
            fold_seed = args.seed + fold_number
            set_seed(fold_seed)

            fold_result = run_single_training(
                args,
                train_items,
                val_items,
                edge_mode,
                device,
                fold_number,
            )

            fold_results.append(fold_result)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results[edge_mode] = aggregate_fold_results(
            fold_results,
            args,
            edge_mode,
        )

    if len(results) > 1:
        combined_path = os.path.join(
            "results",
            f"{args.dataset}_all_ablation_{args.folds}fold_metrics.json",
        )

        with open(
            combined_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                results,
                file,
                indent=4,
            )

        print(
            f"\n📊 All {len(results)} ablation summaries saved to {combined_path}"
        )

        print("\n================ CROSS-VALIDATED ABLATION SUMMARY ================")
        for edge_mode, summary in results.items():
            print(
                f"{edge_mode:15s} | "
                f"ACC {summary['mean']['accuracy']:.4f} ± {summary['std']['accuracy']:.4f} | "
                f"F1 {summary['mean']['f1']:.4f} ± {summary['std']['f1']:.4f} | "
                f"AUC {summary['mean']['auc']:.4f} ± {summary['std']['auc']:.4f}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["qemu", "ffmpeg"],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of stratified CV folds (default: 5).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to create the shared CV folds (default: 42).",
    )

    parser.add_argument(
        "--edges",
        type=str,
        default="all",
        choices=VALID_EDGE_MODES,
        help=(
            "all runs all 7 ablations; otherwise run one concrete "
            "edge configuration"
        ),
    )

    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be >= 2.")

    run_experiments(args)


if __name__ == "__main__":
    main()
