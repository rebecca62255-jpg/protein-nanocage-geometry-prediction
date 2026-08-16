"""
Build the residue-level graph dataset for Model 3.

Nodes:
    Individual amino acid residues from the representative chain

Node features:
    1280-dimensional per-residue ESM2 embeddings

Edges:
    Residue pairs within the same chain whose Cα-Cα distance
    is below 8 Å
"""

import csv
import json
from pathlib import Path

import numpy as np
from Bio import PDB


# ============================================================
# 1. File paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PDB_DIR = ROOT / "data" / "pdb_files_all"
EMB_DIR = ROOT / "data" / "esm_residue_embs"
FEATURE_MATRIX = ROOT / "data" / "feature_matrix.csv"
OUTPUT = ROOT / "data" / "graph_dataset_residue.json"

CONTACT_THRESHOLD = 8.0
MAX_SEQ_LEN = 512


# ============================================================
# 2. Load feature matrix
# ============================================================

print("Loading feature matrix...")

pdb_info = {}

with open(FEATURE_MATRIX) as f:
    reader = csv.DictReader(f)

    for row in reader:

        pdb_id = row["pdb_id"].strip()
        seq_id = row["seq_id"].strip()

        if pdb_id not in pdb_info:

            pdb_info[pdb_id] = {
                "seq_id": seq_id,
                "t_number": row["t_number"].strip(),
                "outer_diameter_A": row["outer_diameter_A"].strip(),
                "split": row["split"].strip()
            }


# ============================================================
# 3. PDB parser
# ============================================================

parser = PDB.PDBParser(
    QUIET=True
)


def get_chain_ca_coords(
    chain,
    max_len=MAX_SEQ_LEN
):
    """
    Extract Cα coordinates from standard residues
    in a single protein chain.
    """

    coords = []

    for residue in chain:

        if residue.get_id()[0] != " ":
            continue

        if "CA" not in residue:
            continue

        coords.append(
            residue["CA"]
            .get_vector()
            .get_array()
        )

    return np.array(
        coords[:max_len]
    )


# ============================================================
# 4. Build residue-level graph dataset
# ============================================================

print("Building residue-level graph dataset...")

graphs = []
skipped = 0


for pdb_id, info in pdb_info.items():

    if not info["t_number"]:
        skipped += 1
        continue


    seq_id = info["seq_id"]


    # --------------------------------------------------------
    # Locate per-residue ESM2 embedding
    # --------------------------------------------------------

    emb_path = (
        EMB_DIR /
        f"{seq_id}.npy"
    )

    if not emb_path.exists():

        emb_path = (
            EMB_DIR /
            f"{seq_id.lower()}.npy"
        )


    if not emb_path.exists():
        skipped += 1
        continue


    # --------------------------------------------------------
    # Load node features
    # --------------------------------------------------------

    node_features = np.load(
        emb_path
    )

    n_nodes = node_features.shape[0]


    if n_nodes < 5:
        skipped += 1
        continue


    # --------------------------------------------------------
    # Identify representative chain
    # --------------------------------------------------------

    try:
        chain_id = seq_id.split(
            "_",
            1
        )[1]

    except IndexError:
        skipped += 1
        continue


    # --------------------------------------------------------
    # Locate PDB structure
    # --------------------------------------------------------

    pdb_path = (
        PDB_DIR /
        f"{pdb_id}.pdb"
    )


    if not pdb_path.exists():
        skipped += 1
        continue


    # --------------------------------------------------------
    # Extract chain
    # --------------------------------------------------------

    try:

        structure = parser.get_structure(
            pdb_id,
            str(pdb_path)
        )

        chain = structure[0][chain_id]

    except Exception:

        skipped += 1
        continue


    # --------------------------------------------------------
    # Extract Cα coordinates
    # --------------------------------------------------------

    ca_coords = get_chain_ca_coords(
        chain
    )


    # --------------------------------------------------------
    # Align embedding length and structural coordinates
    # --------------------------------------------------------

    if len(ca_coords) != n_nodes:

        n_nodes = min(
            len(ca_coords),
            n_nodes
        )

        ca_coords = ca_coords[:n_nodes]

        node_features = (
            node_features[:n_nodes]
        )


    if n_nodes < 5:
        skipped += 1
        continue


    # --------------------------------------------------------
    # Build residue-residue edges
    # --------------------------------------------------------

    diff = (
        ca_coords[:, None, :]
        -
        ca_coords[None, :, :]
    )

    dist = np.linalg.norm(
        diff,
        axis=-1
    )


    idx_i, idx_j = np.where(
        (dist < CONTACT_THRESHOLD)
        &
        (dist > 0)
    )


    edges = [
        [int(i), int(j)]
        for i, j in zip(
            idx_i,
            idx_j
        )
    ]


    # --------------------------------------------------------
    # Store graph
    # --------------------------------------------------------

    graphs.append({

        "pdb_id": pdb_id,

        "n_nodes": n_nodes,

        "node_features":
            node_features.tolist(),

        "edges": edges,

        "t_number":
            info["t_number"],

        "outer_diameter_A":
            info["outer_diameter_A"],

        "split":
            info["split"]
    })


# ============================================================
# 5. Save dataset
# ============================================================

print(
    f"Completed: {len(graphs)} graphs, "
    f"skipped {skipped} entries"
)


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT,
    "w"
) as f:

    json.dump(
        graphs,
        f
    )


print(
    f"Saved residue-level graph dataset to:\n"
    f"{OUTPUT}"
)
