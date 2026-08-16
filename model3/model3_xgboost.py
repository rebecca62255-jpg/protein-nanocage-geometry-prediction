"""
Model 3 XGBoost baseline:
使用與 chain-level GAT 相同的資料切分，
僅使用 1280-dimensional ESM2 embedding 預測 T-number，
不使用 chain count、graph connectivity 或其他 structural features，
作為 chain-level GAT 的 non-graph baseline。
"""

import json
import numpy as np
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier


# ============================================================
# File path
# ============================================================

GRAPH_FILE = "/nobackup/rmgl20/dissertation/scripts/graph_dataset.json"


# ============================================================
# Load graph dataset
# ============================================================

print("讀取圖資料集...")

with open(GRAPH_FILE) as f:
    graphs = json.load(f)


# ============================================================
# Merge rare T-number classes
# ============================================================

train_labels = [
    g["t_number"]
    for g in graphs
    if g["split"] == "train" and g["t_number"]
]

train_counts = Counter(train_labels)

# 與 chain-level GAT 保持一致：
# training set 中樣本數 < 5 的 T-number 合併為 other
rare = {
    t for t, count in train_counts.items()
    if count < 5
}

print(f"合併為 other 的類別: {rare}")


def merge_rare(label):
    return "other" if label in rare else label


# ============================================================
# Label encoding
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

print(f"類別數: {n_classes}")
print(f"類別: {list(le.classes_)}")


# ============================================================
# Prepare train / validation / test data
# ============================================================

X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []


for g in graphs:

    if not g["t_number"]:
        continue

    # --------------------------------------------------------
    # Input feature:
    # 只使用代表鏈的 1280-dimensional ESM2 embedding
    # 不加入 n_chains
    # --------------------------------------------------------

    emb = g["node_features"]
    feat = emb

    # Target
    label = merge_rare(g["t_number"])
    y = le.transform([label])[0]


    # Use exactly the same split stored in graph_dataset.json
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
# Convert to NumPy arrays
# ============================================================

X_train = np.array(X_train, dtype=np.float32)
X_val = np.array(X_val, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)


print(
    f"Train: {len(X_train)}, "
    f"Val: {len(X_val)}, "
    f"Test: {len(X_test)}"
)

print(f"Feature dimension: {X_train.shape[1]}")


# ============================================================
# XGBoost classifier
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
# Training
# ============================================================

print("\n開始訓練 XGBoost...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)


# ============================================================
# Test evaluation
# ============================================================

preds = model.predict(X_test)

acc = accuracy_score(
    y_test,
    preds
)

macro_f1 = f1_score(
    y_test,
    preds,
    average="macro"
)


print("\n===== Test Results =====")

print(f"Test Accuracy: {acc:.4f}")
print(f"Macro F1: {macro_f1:.4f}")


# ============================================================
# Classification report
# ============================================================

labels_in_test = sorted(
    set(y_test)
)

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