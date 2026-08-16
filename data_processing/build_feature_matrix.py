"""
Build the final feature matrix.

Combines:
    - 1280-dimensional ESM2 embeddings
    - dataset split labels
    - T-number
    - outer diameter
    - optional interface features

Output:
    data/feature_matrix.csv

Note:
    Interface features are included when interface_features.csv
    is available. They are not required by the final Model 1 or
    Model 2 implementations.
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

INTERFACE_CSV = ROOT / "data" / "interface_features.csv"

VIPERDB_CSV = ROOT / "data" / "viperdb_filtered_3.5A.csv"

SPLIT_JSON = ROOT / "data" / "dataset_split.json"

OUTPUT_CSV = ROOT / "data" / "feature_matrix.csv"


# ============================================================
# 2. Load dataset split
# ============================================================

print("Loading dataset split...")

with open(SPLIT_JSON) as f:
    split = json.load(f)


# Map each sequence ID to its dataset split
id_to_split = {}

for split_name, ids in split.items():

    for seq_id in ids:
        id_to_split[seq_id] = split_name


print(
    f"Loaded split information for "
    f"{len(id_to_split)} sequence IDs"
)


# ============================================================
# 3. Load optional interface features
# ============================================================

interface = {}


if INTERFACE_CSV.exists():

    print("Loading interface features...")

    with open(INTERFACE_CSV) as f:

        reader = csv.DictReader(f)

        for row in reader:

            pdb_id = (
                row["pdb_id"]
                .strip()
                .lower()
            )

            interface[pdb_id] = {

                "contact_residues":
                    row["contact_residues"],

                "rotation_angle_deg":
                    row["rotation_angle_deg"],

                "bsa_approx":
                    row["bsa_approx"]
            }


    print(
        f"Loaded interface features for "
        f"{len(interface)} PDB entries"
    )


else:

    print(
        "interface_features.csv not found. "
        "Interface feature columns will be left empty."
    )


# ============================================================
# 4. Load geometry labels
# ============================================================

print("Loading VIPERdb labels...")

labels = {}


with open(VIPERDB_CSV) as f:

    reader = csv.DictReader(f)

    for row in reader:

        pdb_id = (
            row["pdb_id"]
            .strip()
            .lower()
        )

        labels[pdb_id] = {

            "t_number":
                row["t_number"],

            "outer_diameter_A":
                row["outer_diameter_A"]
        }


print(
    f"Loaded labels for "
    f"{len(labels)} PDB entries"
)


# ============================================================
# 5. Create feature matrix
# ============================================================

print("Loading ESM2 embeddings...")


OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


with h5py.File(
    EMBEDDINGS_H5,
    "r"
) as h5f:

    seq_ids = list(
        h5f.keys()
    )

    print(
        f"Found {len(seq_ids)} sequence embeddings"
    )


    rows_written = 0
    skipped = 0


    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as out:


        # ----------------------------------------------------
        # Define output columns
        # ----------------------------------------------------

        emb_cols = [
            f"emb_{i}"
            for i in range(1280)
        ]


        fieldnames = [

            "seq_id",
            "pdb_id",
            "split",

            "contact_residues",
            "rotation_angle_deg",
            "bsa_approx",

            "t_number",
            "outer_diameter_A"

        ] + emb_cols


        writer = csv.DictWriter(
            out,
            fieldnames=fieldnames
        )

        writer.writeheader()


        # ----------------------------------------------------
        # Combine features and labels
        # ----------------------------------------------------

        for seq_id in seq_ids:


            pdb_id = (
                seq_id
                .split("_")[0]
                .lower()
            )


            # Sequence must belong to one of the
            # train / validation / test splits.
            if seq_id not in id_to_split:

                skipped += 1
                continue


            # ESM2 embedding
            emb = h5f[seq_id][:]


            # Optional interface features
            iface = interface.get(
                pdb_id,
                {}
            )


            # Geometry labels
            lab = labels.get(
                pdb_id,
                {}
            )


            row = {

                "seq_id":
                    seq_id,

                "pdb_id":
                    pdb_id,

                "split":
                    id_to_split[seq_id],

                "contact_residues":
                    iface.get(
                        "contact_residues",
                        ""
                    ),

                "rotation_angle_deg":
                    iface.get(
                        "rotation_angle_deg",
                        ""
                    ),

                "bsa_approx":
                    iface.get(
                        "bsa_approx",
                        ""
                    ),

                "t_number":
                    lab.get(
                        "t_number",
                        ""
                    ),

                "outer_diameter_A":
                    lab.get(
                        "outer_diameter_A",
                        ""
                    )
            }


            # Add 1280 ESM2 dimensions
            for i, val in enumerate(emb):

                row[f"emb_{i}"] = round(
                    float(val),
                    6
                )


            writer.writerow(row)

            rows_written += 1


# ============================================================
# 6. Summary
# ============================================================

print(
    f"Finished. Wrote {rows_written} rows "
    f"and skipped {skipped} sequences."
)

print(
    f"Feature matrix saved to:\n"
    f"{OUTPUT_CSV}"
)
