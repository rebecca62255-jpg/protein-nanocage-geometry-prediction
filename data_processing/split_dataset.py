"""
Split the dataset by CD-HIT clusters.

Train / Validation / Test = 70% / 15% / 15%

Sequences belonging to the same CD-HIT cluster are kept
in the same dataset split to reduce sequence similarity
between training, validation, and test sets.
"""

import json
import random
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CLSTR_FILE = ROOT / "data" / "sequences_nr.fasta.clstr"
OUTPUT = ROOT / "data" / "dataset_split.json"


# ============================================================
# 2. Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)


# ============================================================
# 3. Read CD-HIT clusters
# ============================================================

print("Loading CD-HIT clusters...")

clusters = []
current_cluster = []


with open(CLSTR_FILE) as f:

    for line in f:

        line = line.strip()

        # Start of a new cluster
        if line.startswith(">Cluster"):

            if current_cluster:
                clusters.append(current_cluster)

            current_cluster = []

        else:

            # Extract sequence ID
            if ">" in line:

                seq_id = (
                    line.split(">")[1]
                    .split("...")[0]
                )

                current_cluster.append(seq_id)


    # Add the final cluster
    if current_cluster:
        clusters.append(current_cluster)


print(
    f"Found {len(clusters)} clusters"
)


# ============================================================
# 4. Shuffle clusters
# ============================================================

random.shuffle(clusters)


# ============================================================
# 5. Split clusters
# ============================================================

n = len(clusters)

n_train = int(
    n * 0.70
)

n_val = int(
    n * 0.15
)


train_clusters = (
    clusters[:n_train]
)

val_clusters = (
    clusters[n_train:n_train + n_val]
)

test_clusters = (
    clusters[n_train + n_val:]
)


# ============================================================
# 6. Expand clusters into sequence IDs
# ============================================================

train_ids = [
    sid
    for cluster in train_clusters
    for sid in cluster
]

val_ids = [
    sid
    for cluster in val_clusters
    for sid in cluster
]

test_ids = [
    sid
    for cluster in test_clusters
    for sid in cluster
]


print(
    f"Train: {len(train_ids)} sequences "
    f"({len(train_clusters)} clusters)"
)

print(
    f"Validation: {len(val_ids)} sequences "
    f"({len(val_clusters)} clusters)"
)

print(
    f"Test: {len(test_ids)} sequences "
    f"({len(test_clusters)} clusters)"
)


# ============================================================
# 7. Save dataset split
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


split_data = {
    "train": train_ids,
    "validation": val_ids,
    "test": test_ids
}


with open(
    OUTPUT,
    "w"
) as f:

    json.dump(
        split_data,
        f,
        indent=2
    )


print(
    f"Finished. Dataset split saved to:\n"
    f"{OUTPUT}"
)
