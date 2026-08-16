"""
Model 2: ESM2 embedding + T-number -> predict outer diameter (regression)

Input:
    1280-dimensional ESM2 embedding
    + 1 encoded T-number
    = 1281 input features

Output:
    Outer diameter (Å)

Evaluation:
    MAE
    R²
    Predicted vs Actual scatter plot
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder


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
MODEL_PATH = ROOT / "outputs" / "best_model2.pt"
FIGURE_PATH = ROOT / "figures" / "model2_predicted_vs_actual.png"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. Collect T-number classes for encoding
# ============================================================

print("Loading data...")

all_t = []

with open(FEATURE_MATRIX) as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["t_number"].strip():
            all_t.append(
                row["t_number"].strip()
            )


le = LabelEncoder()
le.fit(all_t)


# ============================================================
# 4. Load train / validation / test data
# ============================================================

X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []


with open(FEATURE_MATRIX) as f:

    reader = csv.DictReader(f)

    for row in reader:

        if not row["outer_diameter_A"].strip():
            continue

        if not row["t_number"].strip():
            continue


        # 1280-dimensional ESM2 embedding
        emb = [
            float(row[f"emb_{i}"])
            for i in range(1280)
        ]


        # Encode T-number as one additional feature
        t_enc = float(
            le.transform(
                [row["t_number"].strip()]
            )[0]
        )


        # Total input dimension = 1281
        features = emb + [t_enc]


        # Regression target
        diameter = float(
            row["outer_diameter_A"]
        )


        split = row["split"].strip()


        if split == "train":

            X_train.append(features)
            y_train.append(diameter)


        elif split == "validation":

            X_val.append(features)
            y_val.append(diameter)


        elif split == "test":

            X_test.append(features)
            y_test.append(diameter)


print(
    f"Train: {len(X_train)}, "
    f"Val: {len(X_val)}, "
    f"Test: {len(X_test)}"
)


# ============================================================
# 5. Convert to tensors
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
    y_train,
    dtype=torch.float32
).unsqueeze(1)

y_val_t = torch.tensor(
    y_val,
    dtype=torch.float32
).unsqueeze(1)

y_test_t = torch.tensor(
    y_test,
    dtype=torch.float32
).unsqueeze(1)


# ============================================================
# 6. Dataset class
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
# 7. DataLoaders
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
# 8. MLP regression model
# ============================================================

class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(1281, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 1)
        )


    def forward(self, x):
        return self.net(x)


# ============================================================
# 9. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using {device}")


# ============================================================
# 10. Model setup
# ============================================================

model = MLP().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.MSELoss()


# ============================================================
# 11. Training
# ============================================================

best_val_mae = float("inf")


for epoch in range(50):

    # ------------------------
    # Training
    # ------------------------

    model.train()

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        preds = model(X_batch)

        loss = criterion(
            preds,
            y_batch
        )

        loss.backward()

        optimizer.step()


    # ------------------------
    # Validation
    # ------------------------

    model.eval()

    val_preds = []
    val_true = []


    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)

            preds = (
                model(X_batch)
                .cpu()
                .numpy()
            )

            val_preds.extend(
                preds.flatten()
            )

            val_true.extend(
                y_batch.numpy().flatten()
            )


    val_mae = mean_absolute_error(
        val_true,
        val_preds
    )


    # Save model with lowest validation MAE
    if val_mae < best_val_mae:

        best_val_mae = val_mae

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch + 1}/50 | "
            f"Val MAE: {val_mae:.2f} | "
            f"Best: {best_val_mae:.2f}"
        )


# ============================================================
# 12. Load best model
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# 13. Test prediction
# ============================================================

all_preds = []
all_labels = []


with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)

        preds = (
            model(X_batch)
            .cpu()
            .numpy()
        )

        all_preds.extend(
            preds.flatten()
        )

        all_labels.extend(
            y_batch.numpy().flatten()
        )


# ============================================================
# 14. Test metrics
# ============================================================

mae = mean_absolute_error(
    all_labels,
    all_preds
)

r2 = r2_score(
    all_labels,
    all_preds
)


print("\n===== Test Results =====")

print(
    f"Test MAE: {mae:.2f} Å"
)

print(
    f"Test R²: {r2:.4f}"
)


# ============================================================
# 15. Predicted vs Actual scatter plot
# ============================================================

actual = np.array(all_labels)
predicted = np.array(all_preds)


fig, ax = plt.subplots(
    figsize=(8, 8)
)


ax.scatter(
    actual,
    predicted,
    alpha=0.7
)


# Ideal prediction line: y = x
min_value = min(
    actual.min(),
    predicted.min()
)

max_value = max(
    actual.max(),
    predicted.max()
)

ax.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)


ax.set_xlabel(
    "Actual Outer Diameter (Å)"
)

ax.set_ylabel(
    "Predicted Outer Diameter (Å)"
)


ax.text(
    0.05,
    0.95,
    f"MAE = {mae:.2f} Å\nR² = {r2:.4f}",
    transform=ax.transAxes,
    verticalalignment="top"
)


plt.tight_layout()


plt.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    f"\nPredicted vs actual plot saved to:\n"
    f"{FIGURE_PATH}"
)
