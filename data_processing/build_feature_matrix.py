"""
Build the final feature matrix for downstream modelling.

Combines:
    - 1280-dimensional ESM2 embeddings
    - train / validation / test split labels
    - T-number
    - outer diameter

Inputs:
    data/esm_embeddings.h5
    data/dataset_split.json
    data/viperdb_filtered_3.5A.csv

Output:
    data/feature_matrix.csv

Final columns:
    seq_id
    pdb_id
    split
    t_number
    outer_diameter
    emb_0 ... emb_1279
"""

import csv
import json
from pathlib import Path

import h5py


# ============================================================
# 1. File paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS_H5 = ROOT / "data" / "esm_embeddings.h5"
VIPERDB_CSV = ROOT / "data" / "viperdb_filtered_3.5A.csv"
SPLIT_JSON = ROOT / "data" / "dataset_split.json"
OUTPUT_CSV = ROOT / "data" / "feature_matrix.csv"


# ============================================================
# 2. Load dataset split
# ============================================================

print("Loading dataset split...")

with SPLIT_JSON.open(encoding="utf-8") as f:
    split = json.load(f)


id_to_split = {}

for split_name, seq_ids in split.items():

    for seq_id in seq_ids:

        if seq_id in id_to_split:
            raise ValueError(
                f"Sequence ID appears in multiple splits: {seq_id}"
            )

        id_to_split[seq_id] = split_name


print(
    f"Loaded split information for "
    f"{len(id_to_split)} sequence IDs"
)


# ============================================================
# 3. Load VIPERdb labels
# ============================================================

print("Loading VIPERdb labels...")

labels = {}


with VIPERDB_CSV.open(encoding="utf-8") as f:

    reader = csv.DictReader(f)

    required_columns = {
        "pdb_id",
        "t_number",
        "outer_diameter",
    }

    if reader.fieldnames is None:
        raise ValueError(
            "VIPERdb CSV has no header."
        )

    missing_columns = (
        required_columns
        - set(reader.fieldnames)
    )

    if missing_columns:
        raise ValueError(
            "Missing required VIPERdb columns: "
            + ", ".join(sorted(missing_columns))
        )


    for row in reader:

        pdb_id = (
            row["pdb_id"]
            .strip()
            .lower()
        )

        labels[pdb_id] = {
            "t_number":
                row["t_number"].strip(),

            "outer_diameter":
                row["outer_diameter"].strip(),
        }


print(
    f"Loaded labels for "
    f"{len(labels)} PDB entries"
)


# ============================================================
# 4. Load embeddings and build feature matrix
# ============================================================

print("Loading ESM2 embeddings...")


OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


missing_split = []
missing_label = []
missing_t_number = []
missing_outer_diameter = []

rows_written = 0


with h5py.File(
    EMBEDDINGS_H5,
    "r"
) as h5f:

    seq_ids = list(
        h5f.keys()
    )


    print(
        f"Found {len(seq_ids)} "
        f"sequence embeddings"
    )


    emb_cols = [
        f"emb_{i}"
        for i in range(1280)
    ]


    fieldnames = [
        "seq_id",
        "pdb_id",
        "split",
        "t_number",
        "outer_diameter",
    ] + emb_cols


    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as out:

        writer = csv.DictWriter(
            out,
            fieldnames=fieldnames
        )

        writer.writeheader()


        for seq_id in seq_ids:

            # ------------------------------------------------
            # Map sequence ID back to PDB ID
            # Example: 3jaz_D -> 3jaz
            # ------------------------------------------------

            pdb_id = (
                seq_id
                .split("_", 1)[0]
                .lower()
            )


            # ------------------------------------------------
            # Check dataset split
            # ------------------------------------------------

            split_name = id_to_split.get(
                seq_id
            )

            if split_name is None:
                missing_split.append(
                    seq_id
                )
                continue


            # ------------------------------------------------
            # Check VIPERdb labels
            # ------------------------------------------------

            lab = labels.get(
                pdb_id
            )

            if lab is None:
                missing_label.append(
                    seq_id
                )
                continue


            t_number = lab[
                "t_number"
            ]

            outer_diameter = lab[
                "outer_diameter"
            ]


            if not t_number:
                missing_t_number.append(
                    seq_id
                )
                continue


            if not outer_diameter:
                missing_outer_diameter.append(
                    seq_id
                )
                continue


            # ------------------------------------------------
            # Load ESM2 embedding
            # ------------------------------------------------

            emb = h5f[
                seq_id
            ][:]


            if len(emb) != 1280:
                raise ValueError(
                    f"{seq_id} has embedding "
                    f"dimension {len(emb)}, "
                    f"expected 1280."
                )


            # ------------------------------------------------
            # Build output row
            # ------------------------------------------------

            row = {
                "seq_id":
                    seq_id,

                "pdb_id":
                    pdb_id,

                "split":
                    split_name,

                "t_number":
                    t_number,

                "outer_diameter":
                    outer_diameter,
            }


            for i, value in enumerate(
                emb
            ):

                row[
                    f"emb_{i}"
                ] = round(
                    float(value),
                    6
                )


            writer.writerow(
                row
            )

            rows_written += 1


# ============================================================
# 5. Validation summary
# ============================================================

print()
print("Feature matrix validation")
print("-------------------------")

print(
    f"Embeddings found: "
    f"{len(seq_ids)}"
)

print(
    f"Split IDs: "
    f"{len(id_to_split)}"
)

print(
    f"Rows written: "
    f"{rows_written}"
)

print(
    f"Missing split: "
    f"{len(missing_split)}"
)

print(
    f"Missing PDB labels: "
    f"{len(missing_label)}"
)

print(
    f"Missing T-number: "
    f"{len(missing_t_number)}"
)

print(
    f"Missing outer diameter: "
    f"{len(missing_outer_diameter)}"
)


# ============================================================
# 6. Strict consistency checks
# ============================================================

if missing_split:
    print(
        "Example missing split IDs:",
        missing_split[:5]
    )


if missing_label:
    print(
        "Example missing label IDs:",
        missing_label[:5]
    )


if missing_t_number:
    print(
        "Example missing T-number IDs:",
        missing_t_number[:5]
    )


if missing_outer_diameter:
    print(
        "Example missing outer diameter IDs:",
        missing_outer_diameter[:5]
    )


if (
    rows_written
    != len(seq_ids)
):

    print()
    print(
        "WARNING: Not all embeddings "
        "were written to the feature matrix."
    )

else:

    print()
    print(
        "All embeddings were mapped "
        "successfully."
    )


print()

print(
    "Feature matrix saved to:"
)

print(
    OUTPUT_CSV
)
