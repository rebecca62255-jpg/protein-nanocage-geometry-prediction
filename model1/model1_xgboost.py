"""
Model 1 baseline: ESM2 embedding + XGBoost prediction of T-number

This model uses the same 1,280-dimensional ESM2 sequence embeddings
and the same train/validation/test split as model1_mlp.py, allowing
a direct comparison between XGBoost and the MLP classifier.
"""

import csv
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
# 1. File paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_MATRIX = ROOT / "data" / "feature_matrix.csv"


# ============================================================
# 2. Load dataset
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
# 3. Merge rare T-number classes
# ============================================================

# Classes with fewer than 10 samples in the training set
# are grouped into the "other" category.

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
# 4. Encode T-number labels
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
# 5. Convert features to NumPy arrays
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


# ============================================================
# 6. XGBoost classifier
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
# 7. Training
# ============================================================

model.fit(
    X_train,
    y_train_enc,
    eval_set=[(X_val, y_val_enc)],
    verbose=False
)


# ============================================================
# 8. Test evaluation
# ============================================================

preds = model.predict(X_test)

acc = accuracy_score(
    y_test_enc,
    preds
)

macro_f1 = f1_score(
    y_test_enc,
    preds,
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
# 9. Classification report
# ============================================================

labels_in_test = sorted(
    set(y_test_enc)
)

target_names = [
    le.classes_[i]
    for i in labels_in_test
]

print("\nClassification Report:")

print(
    classification_report(
        y_test_enc,
        preds,
        labels=labels_in_test,
        target_names=target_names,
        zero_division=0
    )
)
