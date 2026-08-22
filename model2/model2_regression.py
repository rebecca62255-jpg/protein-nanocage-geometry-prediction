"""
Model 2: ESM2 embedding + T-number -> predict outer diameter (regression)

Input:
    1280-dimensional ESM2 embedding
    + one-hot encoded T-number

Output:
    Outer diameter

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
from sklearn.preprocessing import OneHotEncoder


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

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. Load raw train / validation / test data
# ============================================================

print("Loading data...")

raw = {
    "train": [],
    "validation": [],
    "test": [],
}


with FEATURE_MATRIX.open(
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    required_columns = {
        "split",
        "t_number",
        "outer_diameter",
    }

    if reader.fieldnames is None:
        raise ValueError(
            "feature_matrix.csv has no header."
        )

    missing_columns = (
        required_columns
        - set(reader.fieldnames)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )


    for row in reader:

        split = row["split"].strip()

        if split not in raw:
            continue


        t_number = (
            row["t_number"]
            .strip()
        )

        diameter = (
            row["outer_diameter"]
            .strip()
        )


        if not t_number:
            continue

        if not diameter:
            continue


        emb = np.array(
            [
                float(
                    row[f"emb_{i}"]
                )
                for i in range(1280)
            ],
            dtype=np.float32
        )


        raw[split].append(
            {
                "embedding":
                    emb,

                "t_number":
                    t_number,

                "diameter":
                    float(diameter),
            }
        )


print(
    f"Train: {len(raw['train'])}, "
    f"Val: {len(raw['validation'])}, "
    f"Test: {len(raw['test'])}"
)


# ============================================================
# 4. Fit T-number encoder using TRAINING data only
# ============================================================

train_t = np.array(
    [
        [item["t_number"]]
        for item in raw["train"]
    ]
)


encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoder.fit(
    train_t
)


print(
    f"T-number categories learned from train: "
    f"{len(encoder.categories_[0])}"
)

print(
    f"Categories: "
    f"{list(encoder.categories_[0])}"
)


# ============================================================
# 5. Build model inputs
# ============================================================

def build_split(items):

    embeddings = np.stack(
        [
            item["embedding"]
            for item in items
        ]
    )


    t_values = np.array(
        [
            [item["t_number"]]
            for item in items
        ]
    )


    t_onehot = encoder.transform(
        t_values
    ).astype(
        np.float32
    )


    X = np.concatenate(
        [
            embeddings,
            t_onehot
        ],
        axis=1
    )


    y = np.array(
        [
            item["diameter"]
            for item in items
        ],
        dtype=np.float32
    )


    return X, y


X_train, y_train = build_split(
    raw["train"]
)

X_val, y_val = build_split(
    raw["validation"]
)

X_test, y_test = build_split(
    raw["test"]
)


INPUT_DIM = X_train.shape[1]


print(
    f"Final input dimension: "
    f"{INPUT_DIM}"
)


# ============================================================
# 6. Convert to tensors
# ============================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
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
# 7. Dataset
# ============================================================

class EmbDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (
            self.X[index],
            self.y[index]
        )


# ============================================================
# 8. DataLoaders
# ============================================================

train_loader = DataLoader(
    EmbDataset(
        X_train,
        y_train_t
    ),
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    EmbDataset(
        X_val,
        y_val_t
    ),
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    EmbDataset(
        X_test,
        y_test_t
    ),
    batch_size=64,
    shuffle=False
)


# ============================================================
# 9. MLP regression model
# ============================================================

class MLP(nn.Module):

    def __init__(
        self,
        input_dim
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                512
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                512,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                256,
                1
            )
        )


    def forward(
        self,
        x
    ):

        return self.net(
            x
        )


# ============================================================
# 10. Device
# ============================================================

if torch.cuda.is_available():

    device = torch.device(
        "cuda"
    )

elif (
    hasattr(
        torch.backends,
        "mps"
    )
    and
    torch.backends.mps.is_available()
):

    device = torch.device(
        "mps"
    )

else:

    device = torch.device(
        "cpu"
    )


print(
    f"Using {device}"
)


# ============================================================
# 11. Model setup
# ============================================================

model = MLP(
    INPUT_DIM
).to(
    device
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


criterion = nn.MSELoss()


# ============================================================
# 12. Training
# ============================================================

best_val_mae = float(
    "inf"
)

best_epoch = 0

MAX_EPOCHS = 100
PATIENCE = 15

epochs_without_improvement = 0


for epoch in range(
    MAX_EPOCHS
):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()


    for (
        X_batch,
        y_batch
    ) in train_loader:

        X_batch = X_batch.to(
            device
        )

        y_batch = y_batch.to(
            device
        )


        optimizer.zero_grad()


        preds = model(
            X_batch
        )


        loss = criterion(
            preds,
            y_batch
        )


        loss.backward()

        optimizer.step()


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()


    val_preds = []
    val_true = []


    with torch.no_grad():

        for (
            X_batch,
            y_batch
        ) in val_loader:

            X_batch = X_batch.to(
                device
            )


            preds = (
                model(
                    X_batch
                )
                .cpu()
                .numpy()
                .flatten()
            )


            val_preds.extend(
                preds
            )

            val_true.extend(
                y_batch
                .numpy()
                .flatten()
            )


    val_mae = mean_absolute_error(
        val_true,
        val_preds
    )


    # --------------------------------------------------------
    # Save model with lowest validation MAE
    # --------------------------------------------------------

    if val_mae < best_val_mae:

        best_val_mae = val_mae

        best_epoch = epoch + 1

        epochs_without_improvement = 0


        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


    else:

        epochs_without_improvement += 1


    if (
        (epoch + 1) % 10 == 0
        or
        epoch == 0
    ):

        print(
            f"Epoch {epoch + 1}/{MAX_EPOCHS} | "
            f"Val MAE: {val_mae:.2f} Å | "
            f"Best: {best_val_mae:.2f} Å"
        )


    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            f"Early stopping at "
            f"epoch {epoch + 1}"
        )

        break


print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation MAE: "
    f"{best_val_mae:.2f} Å"
)


# ============================================================
# 13. Load best model
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# 14. Test prediction
# ============================================================

all_preds = []
all_labels = []


with torch.no_grad():

    for (
        X_batch,
        y_batch
    ) in test_loader:

        X_batch = X_batch.to(
            device
        )


        preds = (
            model(
                X_batch
            )
            .cpu()
            .numpy()
            .flatten()
        )


        all_preds.extend(
            preds
        )

        all_labels.extend(
            y_batch
            .numpy()
            .flatten()
        )


# ============================================================
# 15. Test metrics
# ============================================================

mae = mean_absolute_error(
    all_labels,
    all_preds
)

r2 = r2_score(
    all_labels,
    all_preds
)


print()
print(
    "===== Final Test Results ====="
)

print(
    f"Test MAE: "
    f"{mae:.2f} Å"
)

print(
    f"Test R²: "
    f"{r2:.4f}"
)


# ============================================================
# 16. Predicted vs Actual scatter plot
# ============================================================

actual = np.array(
    all_labels
)

predicted = np.array(
    all_preds
)


fig, ax = plt.subplots(
    figsize=(8, 8)
)


ax.scatter(
    actual,
    predicted,
    alpha=0.7
)


min_value = min(
    actual.min(),
    predicted.min()
)

max_value = max(
    actual.max(),
    predicted.max()
)


ax.plot(
    [
        min_value,
        max_value
    ],
    [
        min_value,
        max_value
    ],
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
    (
        f"MAE = {mae:.2f} Å\n"
        f"R² = {r2:.4f}"
    ),
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


print()
print(
    "Predicted vs actual plot "
    "saved to:"
)

print(
    FIGURE_PATH
)

print()
print(
    "Best model saved to:"
)

print(
    MODEL_PATH
)
