"""
Model 3（殘基層級）: GAT 預測 T 值
節點 = 代表鏈中的每個殘基，特徵 = ESM2 per-residue embedding（1280維，各節點不同）
邊 = Cα距離 < 8Å 的殘基對
"""
import json
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

GRAPH_FILE = "/nobackup/rmgl20/dissertation/scripts/graph_dataset_residue.json"

print("讀取圖資料集...")
with open(GRAPH_FILE) as f:
    graphs = json.load(f)

train_labels = [g["t_number"] for g in graphs if g["split"] == "train" and g["t_number"]]
train_counts = Counter(train_labels)
rare = {t for t, count in train_counts.items() if count < 5}
print(f"合併為 other 的類別: {rare}")

def merge_rare(label):
    return "other" if label in rare else label

all_labels = [g["t_number"] for g in graphs if g["t_number"]]
le = LabelEncoder()
le.fit([merge_rare(l) for l in all_labels])
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

model = GAT(1280, 256, n_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0
for epoch in range(100):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds = out.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += len(batch.y)
    val_acc = correct / total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "/nobackup/rmgl20/dissertation/scripts/best_model3_residue.pt")
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/100 | Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f}")

model.load_state_dict(torch.load("/nobackup/rmgl20/dissertation/scripts/best_model3_residue.pt"))
model.eval()
all_preds, all_labels_enc = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels_enc.extend(batch.y.cpu().numpy())

acc = accuracy_score(all_labels_enc, all_preds)
macro_f1 = f1_score(all_labels_enc, all_preds, average='macro')
print(f"\nTest Accuracy: {acc:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

labels_in_test = sorted(set(all_labels_enc))
target_names = [le.classes_[i] for i in labels_in_test]
print("\nClassification Report:")
print(classification_report(all_labels_enc, all_preds, labels=labels_in_test, target_names=target_names))
