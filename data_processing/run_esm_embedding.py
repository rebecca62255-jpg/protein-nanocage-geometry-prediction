"""
用 ESM2 對每條序列跑 embedding，做 mean pooling 後存成 HDF5
輸入：sequences_nr.fasta（821條序列）
輸出：esm_embeddings.h5
"""

import torch
import esm
import h5py
from Bio import SeqIO

FASTA_PATH = "/nobackup/rmgl20/dissertation/scripts/sequences_nr.fasta"
OUTPUT_H5 = "/nobackup/rmgl20/dissertation/scripts/esm_embeddings.h5"

# 載入模型
print("Loading ESM2 model...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.eval()

if torch.cuda.is_available():
    model = model.cuda()
    print("Using GPU")
else:
    print("Using CPU")

batch_converter = alphabet.get_batch_converter()

# 讀取序列
records = list(SeqIO.parse(FASTA_PATH, "fasta"))
print(f"Found {len(records)} sequences")

with h5py.File(OUTPUT_H5, "w") as h5f:
    for i, record in enumerate(records):
        seq_id = record.id
        seq = str(record.seq)

        # 截斷過長序列（ESM2 最大 1022）
        if len(seq) > 1022:
            seq = seq[:1022]

        data = [(seq_id, seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])

        # 取第33層的 embedding，去掉 BOS/EOS token
        token_embeddings = results["representations"][33][0, 1:len(seq)+1]

        # mean pooling → (1280,)
        mean_embedding = token_embeddings.mean(dim=0).cpu().numpy()

        h5f.create_dataset(seq_id, data=mean_embedding)

        if (i + 1) % 50 == 0 or (i + 1) == len(records):
            print(f"[{i+1}/{len(records)}] {seq_id} done")

print(f"完成！embeddings 存在 {OUTPUT_H5}")
