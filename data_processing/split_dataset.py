"""
Split CD-HIT representative sequences into
training, validation, and test sets.

Input:
    data/sequences_nr.fasta.clstr

Output:
    data/dataset_split.json

CD-HIT was performed at 90% sequence identity.
Only the representative sequence from each cluster is used
for downstream ESM2 embedding and modelling.

Train / Validation / Test = 70% / 15% / 15%
Random seed = 42 for reproducibility.
"""

import json
import random
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CLSTR_FILE = ROOT / "data" / "sequences_nr.fasta.clstr"
OUTPUT_FILE = ROOT / "data" / "dataset_split.json"


# ============================================================
# 2. Settings
# ============================================================

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# 3. Read CD-HIT representative sequences
# ============================================================

print("Loading CD-HIT clusters...")

representatives = []


with CLSTR_FILE.open() as f:

    for line in f:

        line = line.strip()

        # Representative sequence is marked with *
        if line.endswith("*") and ">" in line:

            seq_id = (
                line.split(">")[1]
                .split("...")[0]
            )

            representatives.append(seq_id)


print(
    f"Found {len(representatives)} "
    f"CD-HIT representative sequences"
)


if not representatives:

    raise RuntimeError(
        "No representative sequences were found "
        "in the CD-HIT cluster file."
    )


# ============================================================
# 4. Shuffle reproducibly
# ============================================================

random.seed(SEED)

random.shuffle(
    representatives
)


# ============================================================
# 5. Split representatives
# ============================================================

total = len(representatives)

n_train = int(
    total * TRAIN_RATIO
)

n_val = int(
    total * VAL_RATIO
)


train_ids = (
    representatives[:n_train]
)

validation_ids = (
    representatives[
        n_train:
        n_train + n_val
    ]
)

test_ids = (
    representatives[
        n_train + n_val:
    ]
)


# ============================================================
# 6. Verify no overlap
# ============================================================

train_set = set(train_ids)
val_set = set(validation_ids)
test_set = set(test_ids)


assert train_set.isdisjoint(val_set)
assert train_set.isdisjoint(test_set)
assert val_set.isdisjoint(test_set)


assert (
    len(train_ids)
    + len(validation_ids)
    + len(test_ids)
    == total
)


# ============================================================
# 7. Save split
# ============================================================

split_data = {
    "train": train_ids,
    "validation": validation_ids,
    "test": test_ids,
}


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


with OUTPUT_FILE.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        split_data,
        f,
        indent=2
    )


# ============================================================
# 8. Summary
# ============================================================

def percentage(n):
    return 100 * n / total


print()

print(
    f"Train: {len(train_ids)} "
    f"({percentage(len(train_ids)):.1f}%)"
)

print(
    f"Validation: {len(validation_ids)} "
    f"({percentage(len(validation_ids)):.1f}%)"
)

print(
    f"Test: {len(test_ids)} "
    f"({percentage(len(test_ids)):.1f}%)"
)

print()

print(
    f"Total: {total}"
)

print(
    "Overlap check: PASSED"
)

print(
    f"Random seed: {SEED}"
)

print()

print(
    "Dataset split saved to:"
)

print(
    OUTPUT_FILE
)
