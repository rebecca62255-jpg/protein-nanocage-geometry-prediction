import csv
import os
import time
import urllib.request

INPUT_CSV = "/nobackup/rmgl20/dissertation/scripts/viperdb_filtered_3.5A.csv"
OUTPUT_DIR = "/nobackup/rmgl20/dissertation/scripts/pdb_files_all"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_CSV) as f:
    reader = csv.DictReader(f)
    entries = list(reader)

total = len(entries)
done = 0
failed = []

for entry in entries:
    pdb_id = entry["pdb_id"].strip().lower()

    # Skip if already downloaded (either format)
    if os.path.exists(os.path.join(OUTPUT_DIR, f"{pdb_id}.pdb")):
        done += 1
        continue
    if os.path.exists(os.path.join(OUTPUT_DIR, f"{pdb_id}.cif")):
        done += 1
        continue

    # Try .pdb first
    downloaded = False
    for ext in ["pdb", "cif"]:
        url = f"https://files.rcsb.org/download/{pdb_id}.{ext}"
        out_path = os.path.join(OUTPUT_DIR, f"{pdb_id}.{ext}")
        try:
            urllib.request.urlretrieve(url, out_path)
            done += 1
            downloaded = True
            if done % 50 == 0:
                print(f"[{done}/{total}] downloaded {pdb_id}.{ext}")
            time.sleep(0.2)
            break
        except Exception:
            continue

    if not downloaded:
        failed.append(pdb_id)
        print(f"FAILED: {pdb_id} - not found in any format")

print(f"Done: {done}/{total}, Failed: {len(failed)}")
if failed:
    print("Failed list:", failed)
