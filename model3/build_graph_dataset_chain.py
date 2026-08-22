"""
Build the chain-level graph dataset for Model 3.

Nodes:
    Protein chains

Node features:
    1280-dimensional representative-chain ESM2 embedding

Edges:
    An edge is created between two chains if at least one pair
    of Cα atoms has a distance below 8 Å.

The resulting graph dataset is used by the chain-level GAT.
"""

import csv
import json
from pathlib import Path
from itertools import combinations

import h5py
import numpy as np
from Bio import PDB


# ============================================================
# 1. File paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PDB_DIR = ROOT / "data" / "pdb_files_all"
EMB_FILE = ROOT / "data" / "esm_embeddings.h5"
FEATURE_MATRIX = ROOT / "data" / "feature_matrix.csv"
SPLIT_FILE = ROOT / "data" / "dataset_split.json"
OUTPUT = ROOT / "data" / "graph_dataset.json"

CONTACT_THRESHOLD = 8.0


# ============================================================
# 2. Load ESM2 embeddings
# ============================================================

print("Loading ESM2 embeddings...")

embeddings = {}

with h5py.File(EMB_FILE, "r") as f:

    for key in f.keys():

        embeddings[key] = np.array(
            f[key]
        )


print(
    f"Loaded {len(embeddings)} embeddings"
)


# ============================================================
# 3. Load feature matrix
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

                "seq_id":
                    seq_id,

                "t_number":
                    row["t_number"].strip(),

                "outer_diameter":
                    row["outer_diameter"].strip(),

                "split":
                    row["split"].strip()
            }


# ============================================================
# 4. Load dataset split
# ============================================================

print("Loading dataset split...")

with open(SPLIT_FILE) as f:

    split_data = json.load(f)


# These sets are retained as a reference to the original
# representative-sequence dataset split.

train_ids = set(
    split_data["train"]
)

val_ids = set(
    split_data["validation"]
)

test_ids = set(
    split_data["test"]
)


print(
    f"Split entries: "
    f"train={len(train_ids)}, "
    f"validation={len(val_ids)}, "
    f"test={len(test_ids)}"
)


# ============================================================
# 5. PDB parser
# ============================================================

parser = PDB.PDBParser(
    QUIET=True
)


# ============================================================
# 6. Extract C-alpha coordinates for each chain
# ============================================================

def get_chain_ca_atoms(structure):
    """
    Extract C-alpha coordinates from each protein chain
    in the first structural model.
    """

    chains = {}

    for model in structure:

        for chain in model:

            ca_atoms = []

            for residue in chain:

                if "CA" in residue:

                    ca_atoms.append(
                        residue["CA"]
                        .get_vector()
                        .get_array()
                    )

            if ca_atoms:

                chains[chain.id] = np.array(
                    ca_atoms
                )

        # Use only the first model
        break

    return chains


# ============================================================
# 7. Determine whether two chains are in contact
# ============================================================

def chains_in_contact(
    ca1,
    ca2,
    threshold=CONTACT_THRESHOLD
):
    """
    Return True if at least one pair of C-alpha atoms
    from two chains is closer than the contact threshold.
    """

    for a1 in ca1:

        for a2 in ca2:

            if np.linalg.norm(
                a1 - a2
            ) < threshold:

                return True

    return False


# ============================================================
# 8. Build chain-level graph dataset
# ============================================================

print("Building chain-level graph dataset...")

graphs = []
skipped = 0


for pdb_id, info in pdb_info.items():

    # Skip entries without T-number
    if not info["t_number"]:
        continue


    seq_id = info["seq_id"]


    # --------------------------------------------------------
    # Check ESM2 embedding
    # --------------------------------------------------------

    if seq_id not in embeddings:

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
    # Parse structure and extract chains
    # --------------------------------------------------------

    try:

        structure = parser.get_structure(
            pdb_id,
            str(pdb_path)
        )

        chains = get_chain_ca_atoms(
            structure
        )

    except Exception:

        skipped += 1
        continue


    chain_ids = list(
        chains.keys()
    )

    n_chains = len(
        chain_ids
    )


    # At least two chains are required
    # to construct an inter-chain graph.
    if n_chains < 2:

        skipped += 1
        continue


    # --------------------------------------------------------
    # Node feature
    # --------------------------------------------------------

    # The representative-chain ESM2 embedding is stored once
    # here. model3_chain.py assigns this same embedding to all
    # chain nodes within the graph.

    node_features = (
        embeddings[seq_id]
        .tolist()
    )


    # --------------------------------------------------------
    # Build chain-chain edges
    # --------------------------------------------------------

    edges = []


    for i, j in combinations(
        range(n_chains),
        2
    ):

        if chains_in_contact(
            chains[chain_ids[i]],
            chains[chain_ids[j]]
        ):

            # Store both directions
            edges.append(
                [i, j]
            )

            edges.append(
                [j, i]
            )


    # --------------------------------------------------------
    # Dataset split
    # --------------------------------------------------------

    if info["split"] == "train":

        split = "train"

    elif info["split"] == "validation":

        split = "validation"

    else:

        split = "test"


    # --------------------------------------------------------
    # Store graph
    # --------------------------------------------------------

    graphs.append({

        "pdb_id":
            pdb_id,

        "n_nodes":
            n_chains,

        "node_features":
            node_features,

        "edges":
            edges,

        "t_number":
            info["t_number"],

        "outer_diameter":
            info["outer_diameter"],

        "split":
            split
    })


# ============================================================
# 9. Save graph dataset
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
    f"Saved chain-level graph dataset to:\n"
    f"{OUTPUT}"
)
