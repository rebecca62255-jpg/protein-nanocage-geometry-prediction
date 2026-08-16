"""
建立最終 feature matrix
ESM2 embeddings + interface features + geometry labels
輸出：feature_matrix.csv
"""

import h5py
import csv
import json
import numpy as np

EMBEDDINGS_H5 = "/nobackup/rmgl20/dissertation/scripts/esm_embeddings.h5"
INTERFACE_CSV = "/nobackup/rmgl20/dissertation/scripts/interface_features.csv"
VIPERDB_CSV = "/nobackup/rmgl20/dissertation/scripts/viperdb_filtered_3.5A.csv"
SPLIT_JSON = "/nobackup/rmgl20/dissertation/scripts/dataset_split.json"
OUTPUT_CSV = "/nobackup/rmgl20/dissertation/scripts/feature_matrix.csv"

# 讀取 split
with open(SPLIT_JSON) as f:
    split = json.load(f)

all_ids = set(split["train"] + split["validation"] + split["test"])

# 讀取 interface features
interface = {}
with open(INTERFACE_CSV) as f:
    for row in csv.DictReader(f):
        pdb_id = row["pdb_id"].strip().lower()
        interface[pdb_id] = {
            "contact_residues": row["contact_residues"],
            "rotation_angle_deg": row["rotation_angle_deg"],
            "bsa_approx": row["bsa_approx"],
        }

# 讀取 geometry labels
labels = {}
with open(VIPERDB_CSV) as f:
    for row in csv.DictReader(f):
        pdb_id = row["pdb_id"].strip().lower()
        labels[pdb_id] = {
            "t_number": row["t_number"],
            "outer_diameter_A": row["outer_diameter_A"],
        }

# 讀取 embeddings 並組合
with h5py.File(EMBEDDINGS_H5, "r") as h5f:
    seq_ids = list(h5f.keys())
    print(f"Embeddings 裡有 {len(seq_ids)} 條序列")

    # 決定 split 標籤
    id_to_split = {}
    for s, ids in split.items():
        for sid in ids:
            id_to_split[sid] = s

    rows_written = 0
    skipped = 0

    with open(OUTPUT_CSV, "w", newline="") as out:
        # 先寫 header
        emb_cols = [f"emb_{i}" for i in range(1280)]
        fieldnames = ["seq_id", "pdb_id", "split",
                      "contact_residues", "rotation_angle_deg", "bsa_approx",
                      "t_number", "outer_diameter_A"] + emb_cols
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        for seq_id in seq_ids:
            pdb_id = seq_id.split("_")[0].lower()

            if seq_id not in id_to_split:
                skipped += 1
                continue

            emb = h5f[seq_id][:]
            iface = interface.get(pdb_id, {})
            lab = labels.get(pdb_id, {})

            row = {
                "seq_id": seq_id,
                "pdb_id": pdb_id,
                "split": id_to_split[seq_id],
                "contact_residues": iface.get("contact_residues", ""),
                "rotation_angle_deg": iface.get("rotation_angle_deg", ""),
                "bsa_approx": iface.get("bsa_approx", ""),
                "t_number": lab.get("t_number", ""),
                "outer_diameter_A": lab.get("outer_diameter_A", ""),
            }
            for i, val in enumerate(emb):
                row[f"emb_{i}"] = round(float(val), 6)

            writer.writerow(row)
            rows_written += 1

print(f"完成！共寫入 {rows_written} 筆，跳過 {skipped} 筆")
print(f"結果存在 {OUTPUT_CSV}")
