"""
Download PDB/mmCIF structure files for all entries
in the filtered VIPERdb dataset.

Input:
    data/viperdb_filtered_3.5A.csv

Output:
    data/pdb_files_all/
"""

import csv
import time
import urllib.request
from pathlib import Path


# ============================================================
# 1. File paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = ROOT / "data" / "viperdb_filtered_3.5A.csv"

OUTPUT_DIR = ROOT / "data" / "pdb_files_all"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Load VIPERdb entries
# ============================================================

print("Loading filtered VIPERdb dataset...")

with open(INPUT_CSV) as f:
    reader = csv.DictReader(f)
    entries = list(reader)


total = len(entries)

done = 0
failed = []


print(f"Found {total} entries")
print("Downloading structure files...")


# ============================================================
# 3. Download PDB/mmCIF files
# ============================================================

for entry in entries:

    pdb_id = (
        entry["pdb_id"]
        .strip()
        .lower()
    )

    if not pdb_id:
        continue


    # --------------------------------------------------------
    # Skip if already downloaded
    # --------------------------------------------------------

    pdb_path = OUTPUT_DIR / f"{pdb_id}.pdb"
    cif_path = OUTPUT_DIR / f"{pdb_id}.cif"


    if pdb_path.exists():

        done += 1
        continue


    if cif_path.exists():

        done += 1
        continue


    # --------------------------------------------------------
    # Try PDB first, then mmCIF
    # --------------------------------------------------------

    downloaded = False


    for ext in ["pdb", "cif"]:

        url = (
            f"https://files.rcsb.org/download/"
            f"{pdb_id}.{ext}"
        )

        out_path = (
            OUTPUT_DIR /
            f"{pdb_id}.{ext}"
        )


        try:

            urllib.request.urlretrieve(
                url,
                out_path
            )

            done += 1
            downloaded = True


            if done % 50 == 0:

                print(
                    f"[{done}/{total}] "
                    f"downloaded {pdb_id}.{ext}"
                )


            time.sleep(0.2)

            break


        except Exception:

            # Remove incomplete file if one was created
            if out_path.exists():
                out_path.unlink()

            continue


    # --------------------------------------------------------
    # Record failed downloads
    # --------------------------------------------------------

    if not downloaded:

        failed.append(pdb_id)

        print(
            f"FAILED: {pdb_id} - "
            f"not found in any format"
        )


# ============================================================
# 4. Summary
# ============================================================

print(
    f"\nDone: {done}/{total}, "
    f"Failed: {len(failed)}"
)


if failed:

    print(
        "Failed list:",
        failed
    )


print(
    f"\nStructure files saved to:\n"
    f"{OUTPUT_DIR}"
)
