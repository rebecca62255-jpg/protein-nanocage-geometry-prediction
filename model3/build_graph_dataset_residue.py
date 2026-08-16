"""
建立殘基層級的 GAT 圖資料集
節點 = 代表鏈中的每個胺基酸殘基，特徵 = ESM2 per-residue embedding（1280維）
邊 = 同一條鏈中，Cα-Cα 距離 < 8Å 的殘基對
"""
import os
import csv
import json
import numpy as np
from Bio import PDB

PDB_DIR = "/nobackup/rmgl20/dissertation/scripts/pdb_files_all"
EMB_DIR = "/nobackup/rmgl20/dissertation/scripts/esm_residue_embs"
FEATURE_MATRIX = "/nobackup/rmgl20/dissertation/scripts/feature_matrix.csv"
OUTPUT = "/nobackup/rmgl20/dissertation/scripts/graph_dataset_residue.json"

CONTACT_THRESHOLD = 8.0
MAX_SEQ_LEN = 512

print("讀取 feature matrix...")
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

parser = PDB.PDBParser(QUIET=True)

def get_chain_ca_coords(chain, max_len=MAX_SEQ_LEN):
    coords = []
    for residue in chain:
        if residue.get_id()[0] != ' ':
            continue
        if "CA" not in residue:
            continue
        coords.append(residue["CA"].get_vector().get_array())
    return np.array(coords[:max_len])

print("建立殘基層級圖資料集...")
graphs = []
skipped = 0

for pdb_id, info in pdb_info.items():
    if not info["t_number"]:
        skipped += 1
        continue

    seq_id = info["seq_id"]
    emb_path = os.path.join(EMB_DIR, f"{seq_id}.npy")
    if not os.path.exists(emb_path):
        emb_path = os.path.join(EMB_DIR, f"{seq_id.lower()}.npy")
    if not os.path.exists(emb_path):
        skipped += 1
        continue

    node_features = np.load(emb_path)
    n_nodes = node_features.shape[0]
    if n_nodes < 5:
        skipped += 1
        continue

    chain_id = seq_id.split("_", 1)[1]
    pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_path):
        skipped += 1
        continue

    try:
        structure = parser.get_structure(pdb_id, pdb_path)
        chain = structure[0][chain_id]
    except Exception:
        skipped += 1
        continue

    ca_coords = get_chain_ca_coords(chain)

    if len(ca_coords) != n_nodes:
        n_nodes = min(len(ca_coords), n_nodes)
        ca_coords = ca_coords[:n_nodes]
        node_features = node_features[:n_nodes]

    if n_nodes < 5:
        skipped += 1
        continue

    diff = ca_coords[:, None, :] - ca_coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    idx_i, idx_j = np.where((dist < CONTACT_THRESHOLD) & (dist > 0))
    edges = [[int(i), int(j)] for i, j in zip(idx_i, idx_j)]

    graphs.append({
        "pdb_id": pdb_id,
        "n_nodes": n_nodes,
        "node_features": node_features.tolist(),
        "edges": edges,
        "t_number": info["t_number"],
        "outer_diameter_A": info["outer_diameter_A"],
        "split": info["split"]
    })

print(f"完成：{len(graphs)} 個圖，跳過 {skipped} 個")

with open(OUTPUT, "w") as f:
    json.dump(graphs, f)

print(f"已儲存到 {OUTPUT}")
