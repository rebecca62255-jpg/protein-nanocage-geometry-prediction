"""
Verify key fields in the filtered VIPERdb dataset.

Checks:
    - Total number of entries
    - Available columns
    - Missing values in key fields
"""

import csv
from pathlib import Path


# ============================================================
# 1. File path
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = ROOT / "data" / "viperdb_filtered_3.5A.csv"


# ============================================================
# 2. Load dataset
# ============================================================

print("Loading filtered VIPERdb dataset...")

with open(INPUT_CSV) as f:

    reader = csv.DictReader(f)

    entries = list(reader)


# ============================================================
# 3. Basic dataset information
# ============================================================

print(f"Total entries: {len(entries)}")


if not entries:
    print("Dataset is empty.")
    raise SystemExit


print(
    f"Columns: {list(entries[0].keys())}\n"
)


# ============================================================
# 4. Check missing values
# ============================================================

fields_to_check = [
    "pdb_id",
    "t_number",
    "resolution",
    "outer_diameter"
]


for field in fields_to_check:

    if field not in entries[0]:

        print(
            f"WARNING: column '{field}' "
            f"does not exist in CSV"
        )

        continue


    missing = [
        entry["pdb_id"]
        for entry in entries
        if not entry[field].strip()
    ]


    print(
        f"{field}: {len(missing)} missing"
    )


    if missing[:5]:

        print(
            f"  examples: {missing[:5]}"
        )
