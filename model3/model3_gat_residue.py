"""
Model 3: Residue-level GAT for T-number prediction

Nodes:
    Individual residues from the representative protein chain

Node features:
    1280-dimensional per-residue ESM2 embeddings

Edges:
    An edge is created between two residues when their
    Cα-Cα distance is below 8 Å.

Note:
    The residue-level graph is constructed from a single
    representative chain and therefore captures intra-chain
    structural information rather than inter-chain contacts.
"""

import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# ============================================================
# 1. Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. File paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

GRAPH_FILE = ROOT / "data" / "graph_dataset_residue.json"
MODEL_FILE = ROOT / "outputs" / "best_model3_residue.pt"

MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. Load graph dataset
# ============================================================

print("Loading residue-level graph dataset...")

with open(GRAPH_FILE) as f:
    graphs = json.load(f)


# ============================================================
# 4. Merge rare T-number classes
# ============================================================

train_labels = [
    g["t_number"]
    for g in graphs
    if g["split"] == "train" and g["t_number"]
]

train_counts = Counter(train_labels)

rare = {
    t for t, count in train_counts.items()
    if count < 5
}

print(f"Classes merged into 'other': {rare}")


def merge_rare(label):
    return "other" if label in rare else label


# ============================================================
# 5. Label encoding
# ============================================================

all_labels = [
    g["t_number"]
    for g in graphs
    if g["t_number"]
]

le = LabelEncoder()

le.fit([
    merge_rare(label)
    for label in all_labels
])

n_classes = len(le.classes_)

print(f"Number of classes: {n_classes}")
print(f"Classes: {list(le.classes_)}")


# ============================================================
# 6. Convert JSON graphs to PyTorch Geometric Data objects
# ============================================================

def make_data(g):

    label = merge_rare(g["t_number"])
    y = le.transform([label])[0]

    # Each residue has its own 1280-dimensional
    # per-residue ESM2 embedding.
    x = torch.tensor(
        g["node_features"],
        dtype=torch.float32
    )

    if g["edges"]:

        edge_index = torch.tensor(
            g["edges"],
            dtype=torch.long
        ).t().contiguous()

    else:

        edge_index = torch.zeros(
            (2, 0),
            dtype=torch.long
        )

    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor(
            y,
            dtype=torch.long
        )
    )


# ============================================================
# 7. Train / validation / test split
# ============================================================

train_data = []
val_data = []
test_data = []

for g in graphs:

    if not g["t_number"]:
        continue

    d = make_data(g)

    if g["split"] == "train":
        train_data.append(d)

    elif g["split"] == "validation":
        val_data.append(d)

    elif g["split"] == "test":
        test_data.append(d)


print(
    f"Train: {len(train_data)}, "
    f"Val: {len(val_data)}, "
    f"Test: {len(test_data)}"
)


train_loader = DataLoader(
    train_data,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=8,
    shuffle=False
)

test_loader = DataLoader(
    test_data,
    batch_size=8,
    shuffle=False
)


# ============================================================
# 8. GAT model
# ============================================================

class GAT(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim,
        n_classes,
        heads=4
    ):

        super().__init__()

        self.conv1 = GATConv(
            in_dim,
            hidden_dim,
            heads=heads,
            dropout=0.3
        )

        self.conv2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            dropout=0.3
        )

        self.classifier = nn.Linear(
            hidden_dim,
            n_classes
        )

        self.dropout = nn.Dropout(0.3)


    def forward(
        self,
        x,
        edge_index,
        batch
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = global_mean_pool(
            x,
            batch
        )

        return self.classifier(x)


# ============================================================
# 9. Training setup
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using {device}")


model = GAT(
    in_dim=1280,
    hidden_dim=256,
    n_classes=n_classes
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)


criterion = nn.CrossEntropyLoss()


# ============================================================
# 10. Training
# ============================================================

best_val_acc = 0.0

for epoch in range(100):

    model.train()

    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        out = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        loss = criterion(
            out,
            batch.y
        )

        loss.backward()

        optimizer.step()


    # ------------------------
    # Validation
    # ------------------------

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch in val_loader:

            batch = batch.to(device)

            out = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            preds = out.argmax(dim=1)

            correct += (
                preds == batch.y
            ).sum().item()

            total += len(batch.y)


    val_acc = correct / total


    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            MODEL_FILE
        )


    if (epoch + 1) % 20 == 0:

        print(
            f"Epoch {epoch + 1}/100 | "
            f"Val Acc: {val_acc:.4f} | "
            f"Best: {best_val_acc:.4f}"
        )


# ============================================================
# 11. Test evaluation
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device
    )
)

model.eval()

all_preds = []
all_labels_enc = []


with torch.no_grad():

    for batch in test_loader:

        batch = batch.to(device)

        out = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        preds = (
            out.argmax(dim=1)
            .cpu()
            .numpy()
        )

        all_preds.extend(preds)

        all_labels_enc.extend(
            batch.y.cpu().numpy()
        )


acc = accuracy_score(
    all_labels_enc,
    all_preds
)

macro_f1 = f1_score(
    all_labels_enc,
    all_preds,
    average="macro"
)


print("\n===== Test Results =====")

print(
    f"Test Accuracy: {acc:.4f}"
)

print(
    f"Macro F1: {macro_f1:.4f}"
)


# ============================================================
# 12. Classification report
# ============================================================

labels_in_test = sorted(
    set(all_labels_enc)
)

target_names = [
    le.classes_[i]
    for i in labels_in_test
]


print("\nClassification Report:")

print(
    classification_report(
        all_labels_enc,
        all_preds,
        labels=labels_in_test,
        target_names=target_names,
        zero_division=0
    )
)
