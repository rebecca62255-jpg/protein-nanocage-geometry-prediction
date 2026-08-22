"""
Build the residue-level graph dataset for Model 3.

Nodes:
    Individual amino acid residues from each representative chain.

Node features:
    1280-dimensional per-residue ESM2 embeddings.

Edges:
    Directed edges between residues whose Cα-Cα distance is
    below 8 Å.

Inputs:
    data/feature_matrix.csv
    data/pdb_files_all/
    data/esm_residue_embs/

Output:
    data/graph_dataset_residue.json
"""

import csv
import json
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser


# ============================================================
# 1. File paths and settings
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PDB_DIR = ROOT / "data" / "pdb_files_all"
EMB_DIR = ROOT / "data" / "esm_residue_embs"
FEATURE_MATRIX = ROOT / "data" / "feature_matrix.csv"
OUTPUT = ROOT / "data" / "graph_dataset_residue.json"

CONTACT_THRESHOLD = 8.0
MAX_SEQ_LEN = 512


# ============================================================
# 2. Standard amino acids
# ============================================================

STANDARD_AA = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


# ============================================================
# 3. Load feature matrix
# ============================================================

print(
    "Loading feature matrix..."
)

entries = []


with FEATURE_MATRIX.open(
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)


    required_columns = {
        "seq_id",
        "pdb_id",
        "t_number",
        "outer_diameter",
        "split",
    }


    if reader.fieldnames is None:

        raise ValueError(
            "feature_matrix.csv has no header."
        )


    missing_columns = (
        required_columns
        - set(reader.fieldnames)
    )


    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )


    for row in reader:

        entries.append(
            {
                "seq_id":
                    row[
                        "seq_id"
                    ].strip(),

                "pdb_id":
                    row[
                        "pdb_id"
                    ]
                    .strip()
                    .lower(),

                "t_number":
                    row[
                        "t_number"
                    ].strip(),

                "outer_diameter":
                    row[
                        "outer_diameter"
                    ].strip(),

                "split":
                    row[
                        "split"
                    ].strip(),
            }
        )


print(
    f"Found {len(entries)} "
    f"representative sequences"
)


# ============================================================
# 4. Structure parsers
# ============================================================

pdb_parser = PDBParser(
    QUIET=True
)

cif_parser = MMCIFParser(
    QUIET=True
)


# ============================================================
# 5. Extract C-alpha coordinates
# ============================================================

def get_chain_ca_coords(
    chain,
    max_len=MAX_SEQ_LEN
):

    coords = []


    for residue in chain:

        # Standard protein residues only
        if residue.get_id()[0] != " ":
            continue


        if (
            residue
            .get_resname()
            .strip()
            not in STANDARD_AA
        ):

            continue


        if "CA" not in residue:

            continue


        coords.append(
            residue[
                "CA"
            ]
            .get_vector()
            .get_array()
        )


        if (
            len(coords)
            >= max_len
        ):

            break


    return np.asarray(
        coords,
        dtype=np.float32
    )


# ============================================================
# 6. Build graphs
# ============================================================

print(
    "Building residue-level graph dataset..."
)


graphs = []

failed = []


for i, info in enumerate(
    entries,
    1
):

    seq_id = info[
        "seq_id"
    ]

    pdb_id = info[
        "pdb_id"
    ]


    # --------------------------------------------------------
    # Check labels
    # --------------------------------------------------------

    if not info[
        "t_number"
    ]:

        failed.append(
            (
                seq_id,
                "missing T-number"
            )
        )

        continue


    # --------------------------------------------------------
    # Identify representative chain
    # --------------------------------------------------------

    try:

        chain_id = (
            seq_id
            .split(
                "_",
                1
            )[1]
        )

    except IndexError:

        failed.append(
            (
                seq_id,
                "invalid sequence ID"
            )
        )

        continue


    # --------------------------------------------------------
    # Load per-residue ESM2 embedding
    # --------------------------------------------------------

    emb_path = (
        EMB_DIR /
        f"{seq_id}.npy"
    )


    if not emb_path.exists():

        failed.append(
            (
                seq_id,
                "residue embedding not found"
            )
        )

        continue


    try:

        node_features = np.load(
            emb_path
        )

    except Exception as e:

        failed.append(
            (
                seq_id,
                f"embedding load error: {e}"
            )
        )

        continue


    if (
        node_features.ndim != 2
        or
        node_features.shape[1] != 1280
    ):

        failed.append(
            (
                seq_id,
                f"invalid embedding shape "
                f"{node_features.shape}"
            )
        )

        continue


    # --------------------------------------------------------
    # Locate and load structure
    # --------------------------------------------------------

    pdb_path = (
        PDB_DIR /
        f"{pdb_id}.pdb"
    )

    cif_path = (
        PDB_DIR /
        f"{pdb_id}.cif"
    )


    try:

        if pdb_path.exists():

            structure = (
                pdb_parser
                .get_structure(
                    pdb_id,
                    str(pdb_path)
                )
            )

        elif cif_path.exists():

            structure = (
                cif_parser
                .get_structure(
                    pdb_id,
                    str(cif_path)
                )
            )

        else:

            failed.append(
                (
                    seq_id,
                    "structure file not found"
                )
            )

            continue


        chain = structure[0][
            chain_id
        ]


    except Exception as e:

        failed.append(
            (
                seq_id,
                f"structure/chain error: {e}"
            )
        )

        continue


    # --------------------------------------------------------
    # Extract C-alpha coordinates
    # --------------------------------------------------------

    ca_coords = (
        get_chain_ca_coords(
            chain
        )
    )


    # --------------------------------------------------------
    # Align node features and coordinates
    # --------------------------------------------------------

    n_embedding = (
        node_features.shape[0]
    )

    n_coords = len(
        ca_coords
    )


    n_nodes = min(
        n_embedding,
        n_coords
    )


    if n_nodes < 5:

        failed.append(
            (
                seq_id,
                f"too few aligned residues "
                f"(embedding={n_embedding}, "
                f"coords={n_coords})"
            )
        )

        continue


    node_features = (
        node_features[
            :n_nodes
        ]
    )

    ca_coords = (
        ca_coords[
            :n_nodes
        ]
    )


    # --------------------------------------------------------
    # Build residue-residue contact edges
    # --------------------------------------------------------

    diff = (
        ca_coords[
            :,
            None,
            :
        ]
        -
        ca_coords[
            None,
            :,
            :
        ]
    )


    dist = np.linalg.norm(
        diff,
        axis=-1
    )


    idx_i, idx_j = np.where(
        (
            dist
            < CONTACT_THRESHOLD
        )
        &
        (
            dist
            > 0
        )
    )


    edges = [
        [
            int(a),
            int(b)
        ]
        for a, b
        in zip(
            idx_i,
            idx_j
        )
    ]


    # --------------------------------------------------------
    # Store graph
    # --------------------------------------------------------

    graphs.append(
        {
            "seq_id":
                seq_id,

            "pdb_id":
                pdb_id,

            "n_nodes":
                int(
                    n_nodes
                ),

            "node_features":
                node_features
                .astype(
                    np.float32
                )
                .tolist(),

            "edges":
                edges,

            "t_number":
                info[
                    "t_number"
                ],

            "outer_diameter":
                info[
                    "outer_diameter"
                ],

            "split":
                info[
                    "split"
                ],
        }
    )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        i % 50 == 0
        or
        i == len(entries)
    ):

        print(
            f"[{i}/{len(entries)}] "
            f"graphs: {len(graphs)} | "
            f"failed: {len(failed)}"
        )


# ============================================================
# 7. Validate split counts
# ============================================================

split_counts = {
    "train": 0,
    "validation": 0,
    "test": 0,
}


for graph in graphs:

    split_name = graph[
        "split"
    ]

    if (
        split_name
        in split_counts
    ):

        split_counts[
            split_name
        ] += 1


# ============================================================
# 8. Save dataset
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


with OUTPUT.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        graphs,
        f
    )


# ============================================================
# 9. Summary
# ============================================================

print()
print(
    f"Completed: "
    f"{len(graphs)} graphs"
)

print(
    f"Failed/skipped: "
    f"{len(failed)}"
)

print(
    f"Train: "
    f"{split_counts['train']}"
)

print(
    f"Validation: "
    f"{split_counts['validation']}"
)

print(
    f"Test: "
    f"{split_counts['test']}"
)


if failed:

    print()
    print(
        "First failed examples:"
    )

    for item in failed[:10]:

        print(
            " ",
            item
        )


print()
print(
    "Saved residue-level graph dataset to:"
)

print(
    OUTPUT
)
