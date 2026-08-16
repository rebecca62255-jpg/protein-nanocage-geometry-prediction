"""
Model 1 baseline: ESM2 embedding + XGBoost 預測 T 值
跟 model1_mlp.py 用同樣的資料處理方式，方便直接比較
"""
import csv
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

FEATURE_MATRIX = "/nobackup/rmgl20/dissertation/scripts/feature_matrix.csv"

print("讀取資料...")

X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []

with open(FEATURE_MATRIX) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not row["t_number"].strip():
            continue
        emb = [float(row[f"emb_{i}"]) for i in range(1280)]
        t = row["t_number"].strip()
        split = row["split"].strip()
        if split == "train":
            X_train.append(emb); y_train.append(t)
        elif split == "validation":
            X_val.append(emb); y_val.append(t)
        elif split == "test":
            X_test.append(emb); y_test.append(t)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

train_counts = Counter(y_train)
rare = {t for t, count in train_counts.items() if count < 10}
print(f"合併為 other 的類別: {rare}")

def merge_rare(labels):
    return ["other" if t in rare else t for t in labels]

y_train = merge_rare(y_train)
y_val = merge_rare(y_val)
y_test = merge_rare(y_test)

le = LabelEncoder()
le.fit(y_train + y_val + y_test)
y_train_enc = le.transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

n_classes = len(le.classes_)
print(f"類別數: {n_classes}, 類別: {list(le.classes_)}")

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)
model.fit(X_train, y_train_enc, eval_set=[(X_val, y_val_enc)], verbose=False)

preds = model.predict(X_test)
acc = accuracy_score(y_test_enc, preds)
macro_f1 = f1_score(y_test_enc, preds, average='macro')
print(f"\nTest Accuracy: {acc:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

labels_in_test = sorted(set(y_test_enc))
target_names = [le.classes_[i] for i in labels_in_test]
print("\nClassification Report:")
print(classification_report(y_test_enc, preds, labels=labels_in_test, target_names=target_names))