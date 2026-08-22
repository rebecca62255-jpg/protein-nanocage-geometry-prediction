"""
Model 3: XGBoost baseline for T-number classification

Input:
    1,280-dimensional representative-chain ESM2 embedding

Output:
    T-number classification

Dataset:
    Uses the same chain-level subset and train/validation/test split
    as the chain-level GAT.

Preprocessing:
    T-number classes with fewer than 10 samples in the training
    set are merged into an "other" category. Label encoding is
    fitted using the training set only.

Model:
    XGBoost classifier using only the ESM2 embedding as input.

The model does not use:
    - chain count
    - graph connectivity
    - edge features
    - other structural features

Purpose:
    This model serves as a non-graph baseline for evaluating whether
    chain-level graph connectivity provides additional predictive
    information beyond the ESM2 sequence representation.

Evaluation:
    Accuracy
    Macro F1
    Classification report
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)
from xgboost import XGBClassifier


# ============================================================
# 1. File path
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

GRAPH_FILE = ROOT / "data" / "graph_dataset.json"


# ============================================================
# 2. Load graph dataset
# ============================================================

print("Loading graph dataset...")

with open(GRAPH_FILE) as f:
    graphs = json.load(f)


# ============================================================
# 3. Merge rare T-number classes
# ============================================================

train_labels = [
    g["t_number"]
    for g in graphs
    if g["split"] == "train" and g["t_number"]
]

train_counts = Counter(train_labels)

# To match the chain-level GAT setup,
# T-number classes with fewer than 10 training samples
# are grouped into the "other" category.

rare = {
    t for t, count in train_counts.items()
    if count < 10
}

print(f"Classes merged into 'other': {rare}")


def merge_rare(label):
    return "other" if label in rare else label


# ============================================================
# 4. Label encoding
# ============================================================

le = LabelEncoder()

le.fit([
    merge_rare(label)
    for label in train_labels
])

n_classes = len(le.classes_)

print(f"Number of classes: {n_classes}")
print(f"Classes: {list(le.classes_)}")


# ============================================================
# 5. Prepare train / validation / test data
# ============================================================

X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []


for g in graphs:

    if not g["t_number"]:
        continue

    # --------------------------------------------------------
    # Input feature:
    # representative-chain 1280-dimensional ESM2 embedding only
    # --------------------------------------------------------

    emb = g["node_features"]
    feat = emb

    # Target label
    label = merge_rare(g["t_number"])
    y = le.transform([label])[0]

    # Use the split stored in graph_dataset.json
    if g["split"] == "train":
        X_train.append(feat)
        y_train.append(y)

    elif g["split"] == "validation":
        X_val.append(feat)
        y_val.append(y)

    elif g["split"] == "test":
        X_test.append(feat)
        y_test.append(y)


# ============================================================
# 6. Convert to NumPy arrays
# ============================================================

X_train = np.array(
    X_train,
    dtype=np.float32
)

X_val = np.array(
    X_val,
    dtype=np.float32
)

X_test = np.array(
    X_test,
    dtype=np.float32
)

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)


print(
    f"Train: {len(X_train)}, "
    f"Val: {len(X_val)}, "
    f"Test: {len(X_test)}"
)

print(
    f"Feature dimension: {X_train.shape[1]}"
)


# ============================================================
# 7. XGBoost classifier
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)


# ============================================================
# 8. Training
# ============================================================

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)


# ============================================================
# 9. Test evaluation
# ============================================================

preds = model.predict(X_test)

acc = accuracy_score(
    y_test,
    preds
)

labels_in_test = sorted(
    set(y_test)
)

macro_f1 = f1_score(
    y_test,
    preds,
    labels=labels_in_test,
    average="macro",
    zero_division=0
)


print("\n===== Test Results =====")

print(
    f"Test Accuracy: {acc:.4f}"
)

print(
    f"Macro F1: {macro_f1:.4f}"
)


# ============================================================
# 10. Classification report
# ============================================================

target_names = [
    le.classes_[i]
    for i in labels_in_test
]


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        preds,
        labels=labels_in_test,
        target_names=target_names,
        zero_division=0
    )
)
