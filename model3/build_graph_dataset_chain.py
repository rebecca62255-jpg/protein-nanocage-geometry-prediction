"""
從 PDB 檔案建立 GAT 圖資料集
節點 = 蛋白質鏈，邊 = 鏈間接觸
"""

import os
import csv
import json
import numpy as np
import h5py
from Bio import PDB
from itertools import combinations

PDB_DIR = "/nobackup/rmgl20/dissertation/scripts/pdb_files_all"
EMB_FILE = "/nobackup/rmgl20/dissertation/scripts/esm_embeddings.h5"
FEATURE_MATRIX = "/nobackup/rmgl20/dissertation/scripts/feature_matrix.csv"
SPLIT_FILE = "/nobackup/rmgl20/dissertation/scripts/dataset_split.json"
OUTPUT = "/nobackup/rmgl20/dissertation/scripts/graph_dataset.json"

CONTACT_THRESHOLD = 8.0

print("讀取 ESM2 embeddings...")
embeddings = {}
with h5py.File(EMB_FILE, "r") as f:
    for key in f.keys():
        embeddings[key] = np.array(f[key])

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

print("讀取 dataset split...")
with open(SPLIT_FILE) as f:
    split_data = json.load(f)

train_ids = set(split_data["train"])
val_ids = set(split_data["validation"])
test_ids = set(split_data["test"])

parser = PDB.PDBParser(QUIET=True)

def get_chain_ca_atoms(structure):
    chains = {}
    for model in structure:
        for chain in model:
            ca_atoms = []
            for residue in chain:
                if "CA" in residue:
                    ca_atoms.append(residue["CA"].get_vector().get_array())
            if ca_atoms:
                chains[chain.id] = np.array(ca_atoms)
        break
    return chains

def chains_in_contact(ca1, ca2, threshold=CONTACT_THRESHOLD):
    for a1 in ca1:
        for a2 in ca2:
            if np.linalg.norm(a1 - a2) < threshold:
                return True
    return False

print("建立圖資料集...")
graphs = []
skipped = 0

for pdb_id, info in pdb_info.items():
    if not info["t_number"]:
        continue
    
    seq_id = info["seq_id"]
    if seq_id not in embeddings:
        skipped += 1
        continue
    
    pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_path):
        skipped += 1
        continue
    
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
        chains = get_chain_ca_atoms(structure)
    except:
        skipped += 1
        continue
    
    chain_ids = list(chains.keys())
    n_chains = len(chain_ids)
    if n_chains < 2:
        skipped += 1
        continue
    
    node_features = embeddings[seq_id].tolist()
    
    edges = []
    for i, j in combinations(range(n_chains), 2):
        if chains_in_contact(chains[chain_ids[i]], chains[chain_ids[j]]):
            edges.append([i, j])
            edges.append([j, i])
    
    if info["split"] == "train":
        split = "train"
    elif info["split"] == "validation":
        split = "validation"
    else:
        split = "test"
    
    graphs.append({
        "pdb_id": pdb_id,
        "n_nodes": n_chains,
        "node_features": node_features,
        "edges": edges,
        "t_number": info["t_number"],
        "outer_diameter_A": info["outer_diameter_A"],
        "split": split
    })

print(f"完成：{len(graphs)} 個圖，跳過 {skipped} 個")

with open(OUTPUT, "w") as f:
    json.dump(graphs, f)

print(f"已儲存到 {OUTPUT}")