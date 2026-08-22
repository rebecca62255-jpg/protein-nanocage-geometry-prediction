from pathlib import Path
"""
Model 3（殘基層級）: GAT 預測 T 值
節點 = 代表鏈中的每個殘基，特徵 = ESM2 per-residue embedding（1280維，各節點不同）
邊 = Cα距離 < 8Å 的殘基對
"""
import json
import time
import numpy as np
import random
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

GRAPH_FILE = str(Path(__file__).resolve().parents[1] / "data" / "graph_dataset_residue.json")

print("讀取圖資料集...")
with open(GRAPH_FILE) as f:
    graphs = json.load(f)

train_labels = [g["t_number"] for g in graphs if g["split"] == "train" and g["t_number"]]
train_counts = Counter(train_labels)
rare = {t for t, count in train_counts.items() if count < 10}
print(f"合併為 other 的類別: {rare}")

def merge_rare(label):
    return "other" if label in rare else label

le = LabelEncoder()
le.fit([merge_rare(l) for l in train_labels])
n_classes = len(le.classes_)
print(f"類別數: {n_classes}, 類別: {list(le.classes_)}")

def make_data(g):
    label = merge_rare(g["t_number"])
    y = le.transform([label])[0]
    x = torch.tensor(g["node_features"], dtype=torch.float32)
    if g["edges"]:
        edge_index = torch.tensor(g["edges"], dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=torch.tensor(y, dtype=torch.long))

train_data, val_data, test_data = [], [], []
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

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)
test_loader = DataLoader(test_data, batch_size=8)

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=0.3)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        return self.classifier(x)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using {device}")

model = GAT(
    1280,
    128,
    n_classes,
    heads=2
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

best_val_f1 = 0
best_val_acc = 0
best_epoch = 0

MAX_EPOCHS = 100
PATIENCE = 15
epochs_without_improvement = 0

for epoch in range(MAX_EPOCHS):

    epoch_start = time.time()
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds = out.argmax(dim=1)

            val_preds.extend(
                preds.cpu().numpy()
            )

            val_labels.extend(
                batch.y.cpu().numpy()
            )

    val_acc = accuracy_score(
        val_labels,
        val_preds
    )

    labels_in_val = sorted(
        set(val_labels)
    )

    val_f1 = f1_score(
        val_labels,
        val_preds,
        labels=labels_in_val,
        average="macro",
        zero_division=0
    )

    improved = (
        val_f1 > best_val_f1
        or
        (
            val_f1 == best_val_f1
            and val_acc > best_val_acc
        )
    )

    if improved:

        best_val_f1 = val_f1
        best_val_acc = val_acc
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            str(
                Path(__file__).resolve().parent
                / "best_model3_residue.pt"
            )
        )

    else:

        epochs_without_improvement += 1


    epoch_seconds = (
        time.time()
        - epoch_start
    )


    print(
        f"Epoch {epoch + 1}/{MAX_EPOCHS} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val Macro F1: {val_f1:.4f} | "
        f"Best Macro F1: {best_val_f1:.4f} | "
        f"Time: {epoch_seconds:.1f}s"
    )


    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            f"Early stopping at epoch "
            f"{epoch + 1} "
            f"(no improvement for "
            f"{PATIENCE} epochs)"
        )

        break


print()
print("===== Selected GAT =====")
print(f"Best Epoch: {best_epoch}")
print(f"Validation Accuracy: {best_val_acc:.4f}")
print(f"Validation Macro F1: {best_val_f1:.4f}")

model.load_state_dict(
    torch.load(
        str(
            Path(__file__).resolve().parent
            / "best_model3_residue.pt"
        ),
        map_location=device
    )
)
model.eval()
all_preds, all_labels_enc = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels_enc.extend(batch.y.cpu().numpy())

acc = accuracy_score(
    all_labels_enc,
    all_preds
)

labels_in_test = sorted(
    set(all_labels_enc)
)

macro_f1 = f1_score(
    all_labels_enc,
    all_preds,
    labels=labels_in_test,
    average="macro",
    zero_division=0
)

print("\n===== Final Test Results =====")
print(f"Test Accuracy: {acc:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
target_names = [le.classes_[i] for i in labels_in_test]
print("\nClassification Report:")
print(classification_report(all_labels_enc, all_preds, labels=labels_in_test, target_names=target_names))
