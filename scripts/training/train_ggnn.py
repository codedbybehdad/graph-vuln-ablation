# ============================================================
# TRAIN_GGNN.PY
# Improved Stable GGNN for Vulnerability Detection
# ============================================================

import os
import json
import time
import random
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, WeightedRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn.aggr import AttentionalAggregation

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import matplotlib.pyplot as plt


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GraphDataset(Dataset):

    def __init__(self, items, edge_mode="all"):

        self.items = items
        self.edge_mode = edge_mode

    def __len__(self):
        return len(self.items)

    def filter_edges(self, data):

        if self.edge_mode == "all":
            return data

        edge_type_map = {
            "ast": 0,
            "cfg": 1,
            "pdg": 2
        }

        selected = []

        for token in self.edge_mode.split("+"):

            token = token.strip().lower()

            if token in edge_type_map:
                selected.append(edge_type_map[token])

        if len(selected) == 0:
            return data

        mask = torch.zeros_like(data.edge_type, dtype=torch.bool)

        for edge_id in selected:
            mask |= (data.edge_type == edge_id)

        if mask.sum() == 0:
            return data

        data.edge_index = data.edge_index[:, mask]
        data.edge_type = data.edge_type[mask]

        return data

    def __getitem__(self, idx):

        item = self.items[idx]

        data = torch.load(
            item["path"],
            weights_only=False
        )

        data = self.filter_edges(data)

        return data


def split_dataset(index_items):

    labels = [item["label"] for item in index_items]

    train_idx, temp_idx = train_test_split(
        range(len(index_items)),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    temp_labels = [labels[i] for i in temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=temp_labels,
        random_state=42
    )

    train_items = [index_items[i] for i in train_idx]
    val_items = [index_items[i] for i in val_idx]
    test_items = [index_items[i] for i in test_idx]

    return train_items, val_items, test_items


class GGNN(nn.Module):

    def __init__(self, in_channels, hidden_dim=128):

        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )

        self.ggnn1 = GatedGraphConv(
            out_channels=hidden_dim,
            num_layers=2
        )

        self.ggnn2 = GatedGraphConv(
            out_channels=hidden_dim,
            num_layers=2
        )

        self.att_gate = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.att_pool = AttentionalAggregation(self.att_gate)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, edge_type, batch):

        x = self.node_encoder(x)

        residual = x
        x = self.ggnn1(x, edge_index)
        x = x + residual
        x = F.relu(x)

        x = F.dropout(x, p=0.2, training=self.training)

        residual = x
        x = self.ggnn2(x, edge_index)
        x = x + residual
        x = F.relu(x)

        graph_repr = self.att_pool(x, batch)

        logits = self.classifier(graph_repr)

        return logits.view(-1)


def compute_pos_weight(items):

    labels = torch.tensor([item["label"] for item in items])

    pos = labels.sum().item()
    neg = len(labels) - pos

    weight = neg / max(pos, 1)
    weight = min(weight, 5.0)

    return torch.tensor([weight], dtype=torch.float)


def train_epoch(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0.0

    for data in loader:

        data = data.to(device)

        optimizer.zero_grad()

        logits = model(
            data.x,
            data.edge_index,
            data.edge_type,
            data.batch
        )

        labels = data.y.float().view(-1)

        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):

    model.eval()

    probs_all = []
    labels_all = []
    logits_all = []

    with torch.no_grad():

        for data in loader:

            data = data.to(device)

            logits = model(
                data.x,
                data.edge_index,
                data.edge_type,
                data.batch
            )

            probs = torch.sigmoid(logits)

            probs_all.extend(probs.cpu().numpy())
            labels_all.extend(data.y.cpu().numpy())
            logits_all.extend(logits.cpu().numpy())

    probs_all = np.array(probs_all)
    labels_all = np.array(labels_all)

    best_acc = 0.0
    best_threshold = 0.5

    for t in np.arange(0.1, 0.91, 0.02):

        current_preds = (probs_all > t).astype(int)

        current_acc = accuracy_score(labels_all, current_preds)

        if current_acc > best_acc:
            best_acc = current_acc
            best_threshold = t

    threshold = best_threshold

    preds = (probs_all > threshold).astype(int)

    if len(np.unique(labels_all)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(labels_all, probs_all)

    metrics = {

        "accuracy": accuracy_score(labels_all, preds),

        "precision": precision_score(labels_all, preds, zero_division=0),

        "recall": recall_score(labels_all, preds, zero_division=0),

        "f1": f1_score(labels_all, preds, zero_division=0),

        "auc": auc,

        "threshold": threshold,

        "logit_mean": float(np.mean(logits_all)),
        "prob_mean": float(np.mean(probs_all)),
        "prob_std": float(np.std(probs_all))
    }

    return metrics


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, choices=["qemu", "ffmpeg"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument(
        "--edges",
        type=str,
        default="all",
        choices=[
            "all","ast","cfg","pdg",
            "ast+cfg","ast+pdg","cfg+pdg","ast+cfg+pdg"
        ]
    )

    args = parser.parse_args()

    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n🚀 Device:", device)

    dataset_index_path = f"data/processed/{args.dataset}_dataset_index.pt"

    if not os.path.exists(dataset_index_path):
        raise FileNotFoundError(f"Dataset index not found: {dataset_index_path}")

    meta = torch.load(dataset_index_path, weights_only=False)

    dataset_items = meta["graphs"]

    train_items, val_items, test_items = split_dataset(dataset_items)

    train_dataset = GraphDataset(train_items, edge_mode=args.edges)
    val_dataset = GraphDataset(val_items, edge_mode=args.edges)
    test_dataset = GraphDataset(test_items, edge_mode=args.edges)

    train_labels = [item["label"] for item in train_items]

    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts

    sample_weights = [class_weights[label] for label in train_labels]

    sampler = WeightedRandomSampler(sample_weights,len(sample_weights),replacement=True)

    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,sampler=sampler)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size)

    sample = train_dataset[0]
    in_channels = sample.x.shape[1]

    model = GGNN(in_channels=in_channels).to(device)

    pos_weight = compute_pos_weight(train_items).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode="max",factor=0.5,patience=4
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    best_auc = 0.0

    train_losses = []
    val_f1s = []
    val_aucs = []

    for epoch in range(args.epochs):

        start_time = time.time()

        train_loss = train_epoch(model,train_loader,optimizer,criterion,device)

        val_metrics = evaluate(model,val_loader,device)

        scheduler.step(val_metrics["auc"])

        train_losses.append(train_loss)
        val_f1s.append(val_metrics["f1"])
        val_aucs.append(val_metrics["auc"])

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        print(f"Val AUC: {val_metrics['auc']:.4f}")
        print(f"Threshold: {val_metrics['threshold']:.2f}")
        print(f"Learning Rate: {current_lr:.6f}")

        if val_metrics["auc"] > best_auc:

            best_auc = val_metrics["auc"]

            torch.save(
                model.state_dict(),
                f"models/best_model_{args.dataset}_{args.edges}.pt"
            )

            print("✅ Best model updated")

    print("\n📥 Loading best model...")

    model.load_state_dict(
        torch.load(
            f"models/best_model_{args.dataset}_{args.edges}.pt",
            weights_only=True
        )
    )

    test_metrics = evaluate(model,test_loader,device)

    print("\n✅ FINAL TEST RESULTS\n")
    print(json.dumps(test_metrics, indent=4))

    result_file = f"results/{args.dataset}_{args.edges}_metrics.json"

    with open(result_file,"w") as f:
        json.dump(test_metrics,f,indent=4)

    print(f"\n📁 Results saved to {result_file}")

    plt.figure()
    plt.plot(train_losses,label="Train Loss")
    plt.plot(val_f1s,label="Val F1")
    plt.plot(val_aucs,label="Val AUC")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title(f"{args.dataset.upper()} - {args.edges}")
    plt.tight_layout()

    plot_file = f"results/{args.dataset}_{args.edges}_training_curve.png"
    plt.savefig(plot_file)

    print(f"📈 Training curve saved to {plot_file}")


if __name__ == "__main__":
    main()
