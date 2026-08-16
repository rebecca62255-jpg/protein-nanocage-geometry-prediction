"""
重新提取 ESM2 per-residue embedding（每個殘基各自一個向量）
輸出：esm_residue_embs/ 資料夾，每條鏈一個 .npy 檔
格式：pdbid_chain.npy, shape = (seq_len, 1280)
"""
import os
import numpy as np
import torch
import esm
from Bio import PDB

AA_MAP = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

PDB_DIR = "/nobackup/rmgl20/dissertation/scripts/pdb_files_all"
OUTPUT_DIR = "/nobackup/rmgl20/dissertation/scripts/esm_residue_embs"
MAX_SEQ_LEN = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("載入 ESM2 模型...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
batch_converter = alphabet.get_batch_converter()
print(f"Using {device}")

def get_chain_sequence(chain):
    seq = []
    for res in chain.get_residues():
        if res.get_id()[0] != ' ':
            continue
        seq.append(AA_MAP.get(res.get_resname().strip(), 'X'))
    return ''.join(seq)

parser = PDB.PDBParser(QUIET=True)
pdb_files = sorted([f for f in os.listdir(PDB_DIR) if f.endswith('.pdb')])
print(f"找到 {len(pdb_files)} 個 PDB 檔案")

for i, pdb_file in enumerate(pdb_files):
    pdb_id = pdb_file.replace('.pdb', '').lower()
    pdb_path = os.path.join(PDB_DIR, pdb_file)
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
        chains = list(structure[0].get_chains())
    except:
        continue

    for chain in chains[:10]:
        chain_id = chain.get_id()
        key = f"{pdb_id}_{chain_id}"
        out_path = os.path.join(OUTPUT_DIR, f"{key}.npy")

        if os.path.exists(out_path):
            continue

        seq = get_chain_sequence(chain)
        if len(seq) < 5:
            continue
        seq = seq[:MAX_SEQ_LEN]

        try:
            _, _, tokens = batch_converter([(key, seq)])
            tokens = tokens.to(device)
            with torch.no_grad():
                results = model(tokens, repr_layers=[33])
            emb = results["representations"][33][0, 1:len(seq)+1].cpu().numpy()
            np.save(out_path, emb.astype(np.float32))
        except Exception as e:
            print(f"  錯誤 {key}: {e}")

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(pdb_files)} 完成")

print(f"完成！已儲存到 {OUTPUT_DIR}")