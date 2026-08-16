"""
Extract per-residue ESM2 embeddings.

Output:
    data/esm_residue_embs/

Each protein chain is saved as a separate .npy file:
    pdbid_chain.npy

Array shape:
    (sequence_length, 1280)
"""

from pathlib import Path

import numpy as np
import torch
import esm

from Bio import PDB


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
    "VAL": "V"
}


# ============================================================
# 2. File paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PDB_DIR = ROOT / "data" / "pdb_files_all"
OUTPUT_DIR = ROOT / "data" / "esm_residue_embs"

MAX_SEQ_LEN = 512

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. Load ESM2
# ============================================================

print("Loading ESM2 model...")

model, alphabet = (
    esm.pretrained.esm2_t33_650M_UR50D()
)

model.eval()

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = model.to(device)

batch_converter = alphabet.get_batch_converter()

print(f"Using {device}")


# ============================================================
# 4. Extract protein sequence from a chain
# ============================================================

def get_chain_sequence(chain):

    seq = []

    for res in chain.get_residues():

        # Only standard residues
        if res.get_id()[0] != " ":
            continue

        aa = AA_MAP.get(
            res.get_resname().strip(),
            "X"
        )

        seq.append(aa)

    return "".join(seq)


# ============================================================
# 5. Read PDB files
# ============================================================

parser = PDB.PDBParser(
    QUIET=True
)

pdb_files = sorted([
    f
    for f in PDB_DIR.iterdir()
    if f.suffix.lower() == ".pdb"
])

print(
    f"Found {len(pdb_files)} PDB files"
)


# ============================================================
# 6. Generate per-residue embeddings
# ============================================================

for i, pdb_path in enumerate(pdb_files):

    pdb_id = pdb_path.stem.lower()

    try:

        structure = parser.get_structure(
            pdb_id,
            str(pdb_path)
        )

        chains = list(
            structure[0].get_chains()
        )

    except Exception as e:

        print(
            f"Failed to read {pdb_id}: {e}"
        )

        continue


    # Process up to the first 10 chains
    for chain in chains[:10]:

        chain_id = chain.get_id()

        key = f"{pdb_id}_{chain_id}"

        out_path = (
            OUTPUT_DIR /
            f"{key}.npy"
        )


        # Skip embeddings that already exist
        if out_path.exists():
            continue


        seq = get_chain_sequence(chain)


        # Skip very short sequences
        if len(seq) < 5:
            continue


        # Limit sequence length for ESM2
        seq = seq[:MAX_SEQ_LEN]


        try:

            _, _, tokens = batch_converter([
                (key, seq)
            ])

            tokens = tokens.to(device)


            with torch.no_grad():

                results = model(
                    tokens,
                    repr_layers=[33]
                )


            # Remove BOS/EOS tokens and retain
            # one 1280-dimensional vector per residue
            emb = (
                results["representations"][33]
                [0, 1:len(seq) + 1]
                .cpu()
                .numpy()
            )


            np.save(
                out_path,
                emb.astype(np.float32)
            )


        except Exception as e:

            print(
                f"Error processing {key}: {e}"
            )


    if (i + 1) % 50 == 0:

        print(
            f"{i + 1}/{len(pdb_files)} completed"
        )


print(
    f"Finished. Embeddings saved to:\n"
    f"{OUTPUT_DIR}"
)
