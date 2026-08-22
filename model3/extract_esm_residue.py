"""
Extract per-residue ESM2 embeddings for the representative
protein chains used in the final dataset.

Input:
    data/feature_matrix.csv
    data/pdb_files_all/

Output:
    data/esm_residue_embs/

Each representative protein chain is saved as:
    pdbid_chain.npy

Array shape:
    (sequence_length, 1280)

Only sequences present in feature_matrix.csv are processed.
Sequences are truncated to a maximum of 512 residues so that
the residue embeddings remain aligned with the graph-building step.
"""

import csv
from pathlib import Path

import numpy as np
import torch
import esm

from Bio.PDB import PDBParser, MMCIFParser


# ============================================================
# 1. Amino-acid mapping
# ============================================================

AA_MAP = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


# ============================================================
# 2. File paths and settings
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_MATRIX = ROOT / "data" / "feature_matrix.csv"
PDB_DIR = ROOT / "data" / "pdb_files_all"
OUTPUT_DIR = ROOT / "data" / "esm_residue_embs"

MAX_SEQ_LEN = 512

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. Load representative sequence IDs
# ============================================================

print("Loading representative sequence IDs...")

representatives = []

with FEATURE_MATRIX.open(
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        seq_id = (
            row["seq_id"]
            .strip()
        )

        pdb_id = (
            row["pdb_id"]
            .strip()
            .lower()
        )

        if not seq_id:
            continue

        try:
            chain_id = seq_id.split(
                "_",
                1
            )[1]

        except IndexError:
            continue

        representatives.append(
            (
                seq_id,
                pdb_id,
                chain_id
            )
        )


print(
    f"Found {len(representatives)} "
    f"representative chains"
)


# ============================================================
# 4. Load ESM2
# ============================================================

print("Loading ESM2 model...")

model, alphabet = (
    esm.pretrained.esm2_t33_650M_UR50D()
)

model.eval()


if torch.cuda.is_available():

    device = torch.device(
        "cuda"
    )

elif (
    hasattr(
        torch.backends,
        "mps"
    )
    and
    torch.backends.mps.is_available()
):

    device = torch.device(
        "mps"
    )

else:

    device = torch.device(
        "cpu"
    )


model = model.to(
    device
)

batch_converter = (
    alphabet.get_batch_converter()
)

print(
    f"Using {device}"
)


# ============================================================
# 5. Structure parsers
# ============================================================

pdb_parser = PDBParser(
    QUIET=True
)

cif_parser = MMCIFParser(
    QUIET=True
)


# ============================================================
# 6. Extract protein sequence from structure chain
# ============================================================

def get_chain_sequence(
    chain
):

    sequence = []

    for residue in chain:

        # Only standard residues
        if residue.get_id()[0] != " ":
            continue

        aa = AA_MAP.get(
            residue
            .get_resname()
            .strip()
        )

        # Keep sequence and coordinates aligned by
        # skipping unsupported/non-standard residues.
        if aa is None:
            continue

        sequence.append(
            aa
        )

    return "".join(
        sequence
    )


# ============================================================
# 7. Generate per-residue embeddings
# ============================================================

written = 0
failed = []


for i, (
    seq_id,
    pdb_id,
    chain_id
) in enumerate(
    representatives,
    1
):

    out_path = (
        OUTPUT_DIR /
        f"{seq_id}.npy"
    )


    # --------------------------------------------------------
    # Skip existing result
    # --------------------------------------------------------

    if out_path.exists():

        written += 1

        continue


    # --------------------------------------------------------
    # Locate structure
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


        # First structural model only
        model_structure = structure[0]

        chain = model_structure[
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
    # Extract chain sequence
    # --------------------------------------------------------

    seq = get_chain_sequence(
        chain
    )


    if len(seq) < 5:

        failed.append(
            (
                seq_id,
                "sequence shorter than 5 residues"
            )
        )

        continue


    seq = seq[
        :MAX_SEQ_LEN
    ]


    # --------------------------------------------------------
    # Generate residue-level ESM2 embedding
    # --------------------------------------------------------

    try:

        _, _, tokens = (
            batch_converter(
                [
                    (
                        seq_id,
                        seq
                    )
                ]
            )
        )


        tokens = tokens.to(
            device
        )


        with torch.no_grad():

            results = model(
                tokens,
                repr_layers=[33]
            )


        emb = (
            results[
                "representations"
            ][33]
            [
                0,
                1:len(seq) + 1
            ]
            .cpu()
            .numpy()
        )


        if (
            emb.shape[0]
            != len(seq)
        ):

            failed.append(
                (
                    seq_id,
                    "embedding length mismatch"
                )
            )

            continue


        np.save(
            out_path,
            emb.astype(
                np.float32
            )
        )


        written += 1


    except Exception as e:

        failed.append(
            (
                seq_id,
                f"ESM2 error: {e}"
            )
        )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        i % 50 == 0
        or
        i == len(representatives)
    ):

        print(
            f"[{i}/{len(representatives)}] "
            f"processed | "
            f"saved: {written} | "
            f"failed: {len(failed)}"
        )


# ============================================================
# 8. Summary
# ============================================================

print()
print(
    f"Finished."
)

print(
    f"Representative chains: "
    f"{len(representatives)}"
)

print(
    f"Embeddings saved: "
    f"{written}"
)

print(
    f"Failed: "
    f"{len(failed)}"
)


if failed:

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
    "Embeddings saved to:"
)

print(
    OUTPUT_DIR
)
