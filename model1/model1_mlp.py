"""
Model 1: Tuned MLP for T-number prediction using ESM2 embeddings

Input:
    1,280-dimensional ESM2 embedding

Output:
    T-number classification

Preprocessing:
    T-number classes with fewer than 10 samples in the training
    set are merged into an "other" category. Labels are encoded
    using training-set classes only. Input features are standardized
    using statistics fitted on the training set only.

Hyperparameter search:
    Eight MLP configurations are evaluated using the validation set.
    The configurations vary in hidden-layer architecture, dropout,
    learning rate, batch size, and use of class-weighted loss.

Training and model selection:
    Each configuration is trained for up to 150 epochs with early
    stopping (patience = 20). The best epoch for each configuration
    is selected primarily by validation Macro F1, with validation
    accuracy used as a tie-breaker. The final MLP configuration is
    selected using the same criterion.

Final evaluation:
    The selected model is evaluated once on the held-out test set.

Evaluation:
    Accuracy
    Macro F1
    Classification report
    Confusion matrix
"""

import csv
import copy
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight


SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed()


ROOT = Path(__file__).resolve().parents[1]

FEATURE_MATRIX = ROOT / "data" / "feature_matrix.csv"
MODEL_PATH = Path(__file__).resolve().parent / "best_model1_new.pt"
FIGURE_PATH = ROOT / "figures" / "model1_confusion_matrix.png"

FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)


print("Loading data...")

X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []

with open(FEATURE_MATRIX) as f:
    reader = csv.DictReader(f)

    for row in reader:
        if not row["t_number"].strip():
            continue

        emb = [
            float(row[f"emb_{i}"])
            for i in range(1280)
        ]

        label = row["t_number"].strip()
        split = row["split"].strip()

        if split == "train":
            X_train.append(emb)
            y_train.append(label)

        elif split == "validation":
            X_val.append(emb)
            y_val.append(label)

        elif split == "test":
            X_test.append(emb)
            y_test.append(label)


print(
    f"Train: {len(X_train)}, "
    f"Val: {len(X_val)}, "
    f"Test: {len(X_test)}"
)


train_counts = Counter(y_train)

rare = {
    label
    for label, count in train_counts.items()
    if count < 10
}

print(f"Classes merged into 'other': {rare}")


def merge_rare(labels):
    return [
        "other" if label in rare else label
        for label in labels
    ]


y_train = merge_rare(y_train)
y_val = merge_rare(y_val)
y_test = merge_rare(y_test)


le = LabelEncoder()
le.fit(y_train)

y_train_enc = le.transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

n_classes = len(le.classes_)

print(f"Number of classes: {n_classes}")
print(f"Classes: {list(le.classes_)}")


scaler = StandardScaler()

X_train = scaler.fit_transform(
    np.array(X_train, dtype=np.float32)
)

X_val = scaler.transform(
    np.array(X_val, dtype=np.float32)
)

X_test = scaler.transform(
    np.array(X_test, dtype=np.float32)
)


X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train_t = torch.tensor(y_train_enc, dtype=torch.long)
y_val_t = torch.tensor(y_val_enc, dtype=torch.long)
y_test_t = torch.tensor(y_test_enc, dtype=torch.long)


class EmbDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


train_dataset = EmbDataset(X_train, y_train_t)
val_dataset = EmbDataset(X_val, y_val_t)
test_dataset = EmbDataset(X_test, y_test_t)


class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dims,
        dropout,
        n_classes
    ):
        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using {device}")


class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_enc),
    y=y_train_enc
)

class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)


CONFIGS = [
    {
        "name": "MLP_A",
        "hidden_dims": [256, 128],
        "dropout": 0.20,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "class_weights": False
    },
    {
        "name": "MLP_B",
        "hidden_dims": [512, 256],
        "dropout": 0.20,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "class_weights": False
    },
    {
        "name": "MLP_C",
        "hidden_dims": [256, 128],
        "dropout": 0.10,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "class_weights": False
    },
    {
        "name": "MLP_D",
        "hidden_dims": [512, 128],
        "dropout": 0.10,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "class_weights": False
    },
    {
        "name": "MLP_E",
        "hidden_dims": [256],
        "dropout": 0.20,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "class_weights": False
    },
    {
        "name": "MLP_F",
        "hidden_dims": [128],
        "dropout": 0.10,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "class_weights": False
    },
    {
        "name": "MLP_G",
        "hidden_dims": [256, 128],
        "dropout": 0.20,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "class_weights": True
    },
    {
        "name": "MLP_H",
        "hidden_dims": [512, 256],
        "dropout": 0.30,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "class_weights": True
    }
]


def train_candidate(config):

    set_seed()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )

    model = MLP(
        input_dim=1280,
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
        n_classes=n_classes
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    if config["class_weights"]:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_tensor
        )
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0

    MAX_EPOCHS = 150
    PATIENCE = 20

    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):

        model.train()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

        model.eval()

        val_preds = []
        val_true = []

        with torch.no_grad():

            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)

                outputs = model(X_batch)

                preds = (
                    outputs
                    .argmax(dim=1)
                    .cpu()
                    .numpy()
                )

                val_preds.extend(preds)
                val_true.extend(y_batch.numpy())

        val_acc = accuracy_score(
            val_true,
            val_preds
        )

        val_f1 = f1_score(
            val_true,
            val_preds,
            average="macro",
            zero_division=0
        )

        improved = (
            val_f1 > best_val_f1
            or
            (
                val_f1 == best_val_f1
                and
                val_acc > best_val_acc
            )
        )

        if improved:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_state = copy.deepcopy(
                model.state_dict()
            )
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    model.load_state_dict(best_state)

    return {
        "model": model,
        "val_acc": best_val_acc,
        "val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "config": config
    }


print("\n===== MLP Hyperparameter Search =====")

candidate_results = []

for config in CONFIGS:

    print(f"\nTraining {config['name']}...")

    result = train_candidate(config)

    candidate_results.append(result)

    print(
        f"{config['name']} | "
        f"Val Accuracy: {result['val_acc']:.4f} | "
        f"Val Macro F1: {result['val_f1']:.4f} | "
        f"Best Epoch: {result['best_epoch']}"
    )


best_result = max(
    candidate_results,
    key=lambda result: (
        result["val_f1"],
        result["val_acc"]
    )
)

best_model = best_result["model"]
best_config = best_result["config"]


print("\n===== Selected MLP =====")

print(f"Configuration: {best_config['name']}")
print(f"Hidden layers: {best_config['hidden_dims']}")
print(f"Dropout: {best_config['dropout']}")
print(f"Learning rate: {best_config['lr']}")
print(f"Weight decay: {best_config['weight_decay']}")
print(f"Batch size: {best_config['batch_size']}")
print(f"Class weights: {best_config['class_weights']}")
print(f"Validation Accuracy: {best_result['val_acc']:.4f}")
print(f"Validation Macro F1: {best_result['val_f1']:.4f}")
print(f"Best Epoch: {best_result['best_epoch']}")


torch.save(
    best_model.state_dict(),
    MODEL_PATH
)


test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

best_model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)

        outputs = best_model(X_batch)

        preds = (
            outputs
            .argmax(dim=1)
            .cpu()
            .numpy()
        )

        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())


test_acc = accuracy_score(
    all_labels,
    all_preds
)

labels_in_test = sorted(
    set(all_labels)
)

test_macro_f1 = f1_score(
    all_labels,
    all_preds,
    labels=labels_in_test,
    average="macro",
    zero_division=0
)


print("\n===== Final Test Results =====")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Macro F1: {test_macro_f1:.4f}")


target_names = [
    le.classes_[i]
    for i in labels_in_test
]


print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_preds,
        labels=labels_in_test,
        target_names=target_names,
        zero_division=0
    )
)


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
    "Model 1: Tuned MLP T-number Classification"
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

print(
    f"\nSelected model saved to:\n"
    f"{MODEL_PATH}"
)
