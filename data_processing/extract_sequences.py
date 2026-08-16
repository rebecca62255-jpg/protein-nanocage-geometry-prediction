"""
Extract protein chain sequences from downloaded PDB/mmCIF structures.

Input:
    data/viperdb_filtered_3.5A.csv
    data/pdb_files_all/

Output:
    data/sequences.fasta

Only the first structural model is used.
Each unique chain is written once.
Chains shorter than 10 amino acids are skipped.
"""

import csv
from pathlib import Path

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import PPBuilder


# ============================================================
# 1. File paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = ROOT / "data" / "viperdb_filtered_3.5A.csv"
PDB_DIR = ROOT / "data" / "pdb_files_all"
OUTPUT_FASTA = ROOT / "data" / "sequences.fasta"


# ============================================================
# 2. Load VIPERdb metadata
# ============================================================

print("Loading filtered VIPERdb dataset...")

with open(INPUT_CSV) as f:
    reader = csv.DictReader(f)

    entries = {
        row["pdb_id"].strip().lower(): row
        for row in reader
    }


print(f"Found {len(entries)} entries")


# ============================================================
# 3. Initialise parsers
# ============================================================

ppb = PPBuilder()

pdb_parser = PDBParser(
    QUIET=True
)

cif_parser = MMCIFParser(
    QUIET=True
)


# ============================================================
# 4. Extract sequences
# ============================================================

written = 0
failed = []


OUTPUT_FASTA.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_FASTA,
    "w"
) as out:

    for pdb_id, meta in entries.items():

        pdb_path = (
            PDB_DIR /
            f"{pdb_id}.pdb"
        )

        cif_path = (
            PDB_DIR /
            f"{pdb_id}.cif"
        )


        # ----------------------------------------------------
        # Load structure
        # ----------------------------------------------------

        if pdb_path.exists():

            try:

                structure = (
                    pdb_parser
                    .get_structure(
                        pdb_id,
                        str(pdb_path)
                    )
                )

            except Exception:

                failed.append(pdb_id)
                continue


        elif cif_path.exists():

            try:

                structure = (
                    cif_parser
                    .get_structure(
                        pdb_id,
                        str(cif_path)
                    )
                )

            except Exception:

                failed.append(pdb_id)
                continue


        else:

            failed.append(pdb_id)
            continue


        # ----------------------------------------------------
        # Extract unique chains from first model
        # ----------------------------------------------------

        seen_chains = set()


        for model in structure:

            for chain in model:

                chain_id = chain.id


                # Skip duplicate chain IDs
                if chain_id in seen_chains:
                    continue

                seen_chains.add(chain_id)


                # Build peptide fragments
                peptides = ppb.build_peptides(
                    chain
                )


                # Concatenate peptide fragments
                full_seq = "".join(
                    str(pp.get_sequence())
                    for pp in peptides
                )


                # Skip very short chains
                if len(full_seq) < 10:
                    continue


                # ------------------------------------------------
                # FASTA header
                # ------------------------------------------------

                header = (
                    f">{pdb_id}_{chain_id} "
                    f"t={meta.get('t_number', '')} "
                    f"resolution={meta.get('resolution', '')}"
                )


                out.write(
                    header + "\n"
                )

                out.write(
                    full_seq + "\n"
                )


                written += 1


            # Use only the first structural model
            break


# ============================================================
# 5. Summary
# ============================================================

print(
    f"Done: wrote {written} sequences"
)

print(
    f"Failed: {len(failed)} entries"
)


if failed:

    print(
        "Failed:",
        failed
    )


print(
    f"Sequences saved to:\n"
    f"{OUTPUT_FASTA}"
)
