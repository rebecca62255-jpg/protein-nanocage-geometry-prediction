"""
按 CD-HIT cluster 切分資料集
Train/Validation/Test = 70/15/15
"""

import random
import json

CLSTR_FILE = "/nobackup/rmgl20/dissertation/scripts/sequences_nr.fasta.clstr"
OUTPUT = "/nobackup/rmgl20/dissertation/scripts/dataset_split.json"

random.seed(42)

# 讀取 cluster
clusters = []
current_cluster = []

with open(CLSTR_FILE) as f:
    for line in f:
        line = line.strip()
        if line.startswith(">Cluster"):
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = []
        else:
            # 抽出序列 ID
            if ">" in line:
                seq_id = line.split(">")[1].split("...")[0]
                current_cluster.append(seq_id)
    if current_cluster:
        clusters.append(current_cluster)

print(f"總共 {len(clusters)} 個 clusters")

# 打亂 cluster 順序
random.shuffle(clusters)

n = len(clusters)
n_train = int(n * 0.70)
n_val = int(n * 0.15)

train_clusters = clusters[:n_train]
val_clusters = clusters[n_train:n_train+n_val]
test_clusters = clusters[n_train+n_val:]

# 展開成序列 ID 列表
train_ids = [sid for c in train_clusters for sid in c]
val_ids = [sid for c in val_clusters for sid in c]
test_ids = [sid for c in test_clusters for sid in c]

print(f"Train: {len(train_ids)} 序列 ({len(train_clusters)} clusters)")
print(f"Validation: {len(val_ids)} 序列 ({len(val_clusters)} clusters)")
print(f"Test: {len(test_ids)} 序列 ({len(test_clusters)} clusters)")

with open(OUTPUT, "w") as f:
    json.dump({"train": train_ids, "validation": val_ids, "test": test_ids}, f, indent=2)

print(f"完成！切分結果存在 dataset_split.json")
