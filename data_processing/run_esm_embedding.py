"""
Generate ESM2 embeddings for protein sequences.

Each sequence is passed through ESM2 and the residue-level
representations are mean-pooled to produce one 1280-dimensional
embedding per sequence.

Input:
    data/sequences_nr.fasta

Output:
    data/esm_embeddings.h5
"""

from pathlib import Path

import torch
import esm
import h5py
from Bio import SeqIO


# ============================================================
# 1. File paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FASTA_PATH = ROOT / "data" / "sequences_nr.fasta"
OUTPUT_H5 = ROOT / "data" / "esm_embeddings.h5"


# ============================================================
# 2. Load ESM2 model
# ============================================================

print("Loading ESM2 model...")

model, alphabet = (
    esm.pretrained.esm2_t33_650M_UR50D()
)

model.eval()


if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = model.to(device)

print(f"Using {device}")


batch_converter = alphabet.get_batch_converter()


# ============================================================
# 3. Load protein sequences
# ============================================================

records = list(
    SeqIO.parse(
        str(FASTA_PATH),
        "fasta"
    )
)

print(
    f"Found {len(records)} sequences"
)


# ============================================================
# 4. Generate ESM2 embeddings
# ============================================================

OUTPUT_H5.parent.mkdir(
    parents=True,
    exist_ok=True
)


with h5py.File(
    OUTPUT_H5,
    "w"
) as h5f:

    for i, record in enumerate(records):

        seq_id = record.id
        seq = str(record.seq)


        # ----------------------------------------------------
        # Truncate sequences longer than ESM2 input limit
        # ----------------------------------------------------

        if len(seq) > 1022:
            seq = seq[:1022]


        # ----------------------------------------------------
        # Convert sequence to ESM2 tokens
        # ----------------------------------------------------

        data = [
            (seq_id, seq)
        ]

        (
            batch_labels,
            batch_strs,
            batch_tokens
        ) = batch_converter(data)


        batch_tokens = batch_tokens.to(device)


        # ----------------------------------------------------
        # Extract layer-33 representations
        # ----------------------------------------------------

        with torch.no_grad():

            results = model(
                batch_tokens,
                repr_layers=[33]
            )


        # Remove BOS/EOS tokens.
        # Shape: (sequence_length, 1280)
        token_embeddings = (
            results["representations"][33]
            [0, 1:len(seq) + 1]
        )


        # ----------------------------------------------------
        # Mean pooling
        # ----------------------------------------------------

        # Produce one 1280-dimensional embedding
        # for each protein sequence.
        mean_embedding = (
            token_embeddings
            .mean(dim=0)
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Save embedding
        # ----------------------------------------------------

        h5f.create_dataset(
            seq_id,
            data=mean_embedding
        )


        if (
            (i + 1) % 50 == 0
            or
            (i + 1) == len(records)
        ):

            print(
                f"[{i + 1}/{len(records)}] "
                f"{seq_id} done"
            )


print(
    f"Finished. Embeddings saved to:\n"
    f"{OUTPUT_H5}"
)
