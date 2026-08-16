"""
Model 1: ESM2 embedding + MLP prediction of T-number

Input:
    1280-dimensional ESM2 embedding

Output:
    T-number classification

Evaluation:
    Accuracy
    Macro F1
    Classification report
    Confusion matrix
"""

import csv
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight


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

FEATURE_MATRIX = ROOT / "data" / "feature_matrix.csv"
MODEL_PATH = ROOT / "outputs" / "best_model1.pt"
FIGURE_PATH = ROOT / "figures" / "model1_confusion_matrix.png"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. Load dataset
# ============================================================

print("Loading data...")

X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []

with open(FEATURE_MATRIX) as f:
    reader = csv.DictReader(f)

    for row in reader:

        # Skip entries without T-number
        if not row["t_number"].strip():
            continue

        # 1280-dimensional ESM2 embedding
        emb = [
            float(row[f"emb_{i}"])
            for i in range(1280)
        ]

        t = row["t_number"].strip()
        split = row["split"].strip()

        if split == "train":
            X_train.append(emb)
            y_train.append(t)

        elif split == "validation":
            X_val.append(emb)
            y_val.append(t)

        elif split == "test":
            X_test.append(emb)
            y_test.append(t)


print(
    f"Train: {len(X_train)}, "
    f"Val: {len(X_val)}, "
    f"Test: {len(X_test)}"
)


# ============================================================
# 4. Merge rare T-number classes
# ============================================================

# Classes with fewer than 10 training samples
# are grouped into "other".

train_counts = Counter(y_train)

rare = {
    t for t, count in train_counts.items()
    if count < 10
}

print(f"Classes merged into 'other': {rare}")


def merge_rare(labels):
    return [
        "other" if t in rare else t
        for t in labels
    ]


y_train = merge_rare(y_train)
y_val = merge_rare(y_val)
y_test = merge_rare(y_test)


# ============================================================
# 5. Encode T-number labels
# ============================================================

le = LabelEncoder()

le.fit(
    y_train +
    y_val +
    y_test
)

y_train_enc = le.transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

n_classes = len(le.classes_)

print(f"Number of classes: {n_classes}")
print(f"Classes: {list(le.classes_)}")


# ============================================================
# 6. Convert data to PyTorch tensors
# ============================================================

X_train = torch.tensor(
    np.array(X_train),
    dtype=torch.float32
)

X_val = torch.tensor(
    np.array(X_val),
    dtype=torch.float32
)

X_test = torch.tensor(
    np.array(X_test),
    dtype=torch.float32
)

y_train_t = torch.tensor(
    y_train_enc,
    dtype=torch.long
)

y_val_t = torch.tensor(
    y_val_enc,
    dtype=torch.long
)

y_test_t = torch.tensor(
    y_test_enc,
    dtype=torch.long
)


# ============================================================
# 7. Dataset class
# ============================================================

class EmbDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ============================================================
# 8. DataLoaders
# ============================================================

train_loader = DataLoader(
    EmbDataset(X_train, y_train_t),
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    EmbDataset(X_val, y_val_t),
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    EmbDataset(X_test, y_test_t),
    batch_size=64,
    shuffle=False
)


# ============================================================
# 9. MLP model
# ============================================================

class MLP(nn.Module):

    def __init__(self, input_dim, n_classes):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 10. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using {device}")


# ============================================================
# 11. Initialise model
# ============================================================

model = MLP(
    input_dim=1280,
    n_classes=n_classes
).to(device)


# ============================================================
# 12. Class weights
# ============================================================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_enc),
    y=y_train_enc
)

class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)


# ============================================================
# 13. Optimiser and loss
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)


# ============================================================
# 14. Training
# ============================================================

best_val_acc = 0.0

for epoch in range(50):

    # ------------------------
    # Training
    # ------------------------

    model.train()

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(
            outputs,
            y_batch
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

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            preds = outputs.argmax(dim=1)

            correct += (
                preds == y_batch
            ).sum().item()

            total += len(y_batch)


    val_acc = correct / total


    # Save best model according to validation accuracy
    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch + 1}/50 | "
            f"Val Acc: {val_acc:.4f} | "
            f"Best Val Acc: {best_val_acc:.4f}"
        )


# ============================================================
# 15. Load best model
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# 16. Test evaluation
# ============================================================

all_preds = []
all_labels = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)

        outputs = model(X_batch)

        preds = (
            outputs
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

        all_preds.extend(preds)

        all_labels.extend(
            y_batch.numpy()
        )


# ============================================================
# 17. Accuracy and Macro F1
# ============================================================

acc = accuracy_score(
    all_labels,
    all_preds
)

macro_f1 = f1_score(
    all_labels,
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
# 18. Classification report
# ============================================================

labels_in_test = sorted(
    set(all_labels)
)

target_names_in_test = [
    le.classes_[i]
    for i in labels_in_test
]

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_preds,
        labels=labels_in_test,
        target_names=target_names_in_test,
        zero_division=0
    )
)


# ============================================================
# 19. Confusion matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_preds,
    labels=labels_in_test
)

display_labels = [
    le.classes_[i]
    for i in labels_in_test
]

fig, ax = plt.subplots(
    figsize=(10, 8)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=display_labels
)

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    colorbar=False
)

ax.set_title(
    "Model 1: T-number Classification"
)

ax.set_xlabel(
    "Predicted T-number"
)

ax.set_ylabel(
    "Actual T-number"
)

plt.tight_layout()

plt.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nConfusion matrix saved to:\n"
    f"{FIGURE_PATH}"
)
