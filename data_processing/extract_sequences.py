import os
import csv
from Bio import SeqIO
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import PPBuilder

INPUT_CSV = "/nobackup/rmgl20/dissertation/scripts/viperdb_filtered_3.5A.csv"
PDB_DIR = "/nobackup/rmgl20/dissertation/scripts/pdb_files_all"
OUTPUT_FASTA = "/nobackup/rmgl20/dissertation/scripts/sequences.fasta"

with open(INPUT_CSV) as f:
    reader = csv.DictReader(f)
    entries = {row["pdb_id"].strip().lower(): row for row in reader}

ppb = PPBuilder()
pdb_parser = PDBParser(QUIET=True)
cif_parser = MMCIFParser(QUIET=True)

written = 0
failed = []

with open(OUTPUT_FASTA, "w") as out:
    for pdb_id, meta in entries.items():
        pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
        cif_path = os.path.join(PDB_DIR, f"{pdb_id}.cif")

        if os.path.exists(pdb_path):
            try:
                structure = pdb_parser.get_structure(pdb_id, pdb_path)
            except Exception as e:
                failed.append(pdb_id)
                continue
        elif os.path.exists(cif_path):
            try:
                structure = cif_parser.get_structure(pdb_id, cif_path)
            except Exception as e:
                failed.append(pdb_id)
                continue
        else:
            failed.append(pdb_id)
            continue

        # Get unique chains from asymmetric unit
        seen_chains = set()
        for model in structure:
            for chain in model:
                chain_id = chain.id
                if chain_id in seen_chains:
                    continue
                seen_chains.add(chain_id)
                peptides = ppb.build_peptides(chain)
                full_seq = "".join(str(pp.get_sequence()) for pp in peptides)
                if len(full_seq) < 10:
                    continue
                header = f">{pdb_id}_{chain_id} t={meta.get('t_number','')} resolution={meta.get('resolution','')}"
                out.write(header + "\n")
                out.write(full_seq + "\n")
                written += 1
            break  # only use first model

print(f"Done: wrote {written} sequences")
print(f"Failed: {len(failed)} entries")
if failed:
    print("Failed:", failed)
