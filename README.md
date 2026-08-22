# Predicting Icosahedral Virus Capsid Properties from Protein Sequence and Structure

This repository contains the code, processed data, model outputs, and figures used for a dissertation project investigating whether protein sequence representations and structural graph representations can be used to predict properties of icosahedral virus capsids.

The project uses capsid metadata obtained from VIPERdb and experimentally determined structures retrieved from the RCSB Protein Data Bank (PDB). Protein sequences are represented using pretrained ESM-2 embeddings, while graph attention networks (GATs) are used to incorporate residue-level and chain-level structural information.

Three modelling tasks are investigated:

1. **Model 1:** Sequence-based T-number classification
2. **Model 2:** Capsid outer-diameter regression
3. **Model 3:** Structure-aware T-number classification

XGBoost models are additionally used as baseline classifiers for Model 1 and for comparison with the chain-level graph model in Model 3.

---

## Repository Structure

```text
.
├── data/
├── data_processing/
├── figures/
├── model1/
├── model2/
├── model3/
├── results/
├── requirements.txt
└── README.md
```

### `data_processing/`

Scripts used to retrieve, preprocess, and construct the sequence-level dataset.

```text
download_viperdb.py
add_resolution.py
verify_entries.py
download_all_pdb.py
extract_sequences.py
run_cdhit.sh
split_dataset.py
run_esm_embedding.py
build_feature_matrix.py
```

### `model1/`

```text
model1_mlp.py
model1_xgboost.py
```

Contains the sequence-based T-number classification models.

### `model2/`

```text
model2_regression.py
```

Contains the neural-network regression model used to predict capsid outer diameter.

### `model3/`

```text
extract_esm_residue.py
build_graph_dataset_residue.py
model3_gat_residue.py
build_graph_dataset_chain.py
model3_chain.py
model3_xgboost.py
```

Contains the residue-level and chain-level structural graph pipelines and the non-graph XGBoost baseline.

### `results/`

Contains text outputs from the final model runs.

### `figures/`

Contains figures generated during final model evaluation.

---

# 1. Dataset Construction

## 1.1 VIPERdb Data

Capsid metadata were obtained from VIPERdb.

The collected information includes the PDB identifier, T-number, capsid outer diameter, and other available structural metadata.

Structural resolution information was added to the retrieved entries, and structures with a resolution of **3.5 Å or better** were retained for subsequent analysis.

The filtered metadata are stored in:

```text
data/viperdb_filtered_3.5A.csv
```

---

## 1.2 Structure Retrieval

Corresponding three-dimensional structures were retrieved from the RCSB Protein Data Bank.

PDB-format structures were used where available, with mmCIF files used when required by the structure-retrieval and residue-level processing pipeline.

Downloaded structural files are stored locally in:

```text
data/pdb_files_all/
```

Because this directory is large, it is not included in the GitHub repository and can be regenerated using the provided structure-download script.

---

## 1.3 Protein Sequence Extraction

Protein sequences were extracted from the downloaded structures.

Only the first structural model was used. Protein chains were converted to amino-acid sequences, and very short sequences were excluded during sequence extraction.

The resulting sequences are stored in:

```text
data/sequences.fasta
```

---

## 1.4 Sequence Redundancy Reduction

To reduce sequence redundancy, the extracted protein sequences were clustered using **CD-HIT at 90% sequence identity**.

The CD-HIT output files are:

```text
data/sequences_nr.fasta
data/sequences_nr.fasta.clstr
```

One representative sequence was retained from each CD-HIT cluster.

This produced **844 non-redundant representative protein sequences** for the final sequence-level dataset.

---

## 1.5 Train, Validation, and Test Split

After sequence redundancy reduction, the 844 representative sequences were randomly divided into training, validation, and test sets using a fixed random seed of **42**.

The split ratio was approximately 70:15:15.

| Split | Number of samples |
|---|---:|
| Training | 590 |
| Validation | 126 |
| Test | 128 |
| **Total** | **844** |

The fixed split is stored in:

```text
data/dataset_split.json
```

The same sequence-level split is used for Model 1, Model 2, and the residue-level Model 3 analysis.

Importantly, CD-HIT clustering is performed **before** dataset splitting. The retained representative sequences, rather than entire clusters containing all original sequences, are subsequently assigned to the train, validation, and test sets.

---

# 2. ESM-2 Protein Sequence Representation

Protein sequences are represented using the pretrained:

```text
esm2_t33_650M_UR50D
```

model from ESM-2.

For the sequence-level models, representations from **layer 33** are extracted for individual residues and mean-pooled across the sequence to obtain a single:

```text
1,280-dimensional
```

embedding for each representative protein chain.

Sequences longer than the supported processing length are truncated to a maximum of **1,022 residues** during sequence-level embedding generation.

The resulting embeddings are stored in:

```text
data/esm_embeddings.h5
```

The final sequence-level feature matrix is stored in:

```text
data/feature_matrix.csv
```

and contains sequence identifiers, PDB identifiers, dataset split assignments, T-number labels, outer-diameter values, and the 1,280 ESM-2 embedding features.

---

# 3. Model 1: Sequence-based T-number Classification

Model 1 investigates whether the T-number of an icosahedral virus capsid can be predicted using protein sequence information alone.

The input consists of the **1,280-dimensional mean-pooled ESM-2 embedding** of each representative protein sequence.

For classification, T-number classes containing fewer than **10 training samples** are merged into an `other` category.

Rare-class determination is based only on the training set.

---

## 3.1 MLP Classifier

The primary Model 1 classifier is a multilayer perceptron (MLP).

Input features are standardized using a `StandardScaler` fitted only on the training set. The same fitted transformation is subsequently applied to the validation and test sets.

Eight MLP configurations are evaluated. The configurations vary in:

- hidden-layer architecture
- dropout
- learning rate
- batch size
- use of class-weighted loss

Each configuration is trained for up to **150 epochs** with an early-stopping patience of **20 epochs**.

The best epoch for each configuration is selected primarily using **validation Macro F1**, with validation accuracy used as a tie-breaker.

The final MLP configuration is selected using the same criterion.

The held-out test set is used only after model selection.

Evaluation metrics include:

- Accuracy
- Macro F1
- Per-class precision
- Per-class recall
- Per-class F1
- Confusion matrix

Run the MLP using:

```bash
python model1/model1_mlp.py
```

---

## 3.2 XGBoost Baseline

An XGBoost classifier is used as a baseline for Model 1.

The XGBoost model uses the same:

- 1,280-dimensional ESM-2 embeddings
- train/validation/test split
- rare-class merging rule

as the MLP classifier.

The XGBoost model uses fixed hyperparameters rather than a separate hyperparameter search.

Run using:

```bash
python model1/model1_xgboost.py
```

---

# 4. Model 2: Capsid Outer-Diameter Regression

Model 2 investigates whether capsid outer diameter can be predicted from protein sequence representation together with T-number information.

The input consists of:

```text
1,280-dimensional ESM-2 embedding
+
one-hot encoded T-number
```

The T-number encoder is fitted using the training set only, with unseen categories handled without fitting information from the validation or test sets.

For the final dataset, the input dimension is:

```text
1,295 features
```

consisting of 1,280 ESM-2 features and 15 T-number features.

The regression network has the following architecture:

```text
Input
  ↓
512
ReLU
Dropout (0.3)
  ↓
256
ReLU
Dropout (0.3)
  ↓
1 output
```

The model is trained using:

```text
Loss: Mean Squared Error
Optimizer: Adam
Learning rate: 0.001
Batch size: 64
Maximum epochs: 100
Early-stopping patience: 15
```

Model selection is based on **validation Mean Absolute Error (MAE)**.

Final evaluation uses:

- Mean Absolute Error (MAE)
- R²

Run using:

```bash
python model2/model2_regression.py
```

---

# 5. Model 3: Structure-aware T-number Classification

Model 3 investigates whether structural information represented as graphs provides additional information for T-number classification.

Two graph representations are evaluated:

1. Residue-level graph
2. Chain-level graph

Graph Attention Networks (GATs) are used for both representations.

For both classification models, T-number classes containing fewer than **10 samples in the corresponding training set** are merged into an `other` category.

---

# 5.1 Residue-level Graph Representation

The residue-level graph represents the three-dimensional organization of residues within the representative protein chain.

### Nodes

Each node represents an individual amino-acid residue.

### Node features

Each node contains its own **1,280-dimensional per-residue ESM-2 embedding**.

Unlike the sequence-level representation used in Model 1 and Model 2, these residue embeddings are **not mean-pooled**.

### Edges

An edge is created between two residues when their Cα–Cα distance is:

```text
< 8 Å
```

Edges are represented in both directions.

Self-edges are excluded during graph construction.

Residue-level sequences and structures are limited to a maximum of **512 residues** to maintain alignment between the per-residue ESM-2 representation and structural coordinates.

The final residue-level graph dataset contains:

| Split | Graphs |
|---|---:|
| Training | 590 |
| Validation | 126 |
| Test | 128 |
| **Total** | **844** |

---

## 5.2 Generating Residue-level Embeddings

Generate per-residue ESM-2 embeddings using:

```bash
python model3/extract_esm_residue.py
```

The embeddings are saved locally in:

```text
data/esm_residue_embs/
```

This directory is large and is therefore not included in the GitHub repository.

---

## 5.3 Building Residue-level Graphs

Construct the residue-level graph dataset using:

```bash
python model3/build_graph_dataset_residue.py
```

The resulting file is:

```text
data/graph_dataset_residue.json
```

This generated graph dataset is several gigabytes in size and is therefore not included in the GitHub repository.

It can be regenerated from the supplied scripts and source data.

---

## 5.4 Residue-level GAT

The residue-level GAT uses the following architecture:

```text
1280-dimensional residue features
        ↓
GATConv
128 hidden units × 2 attention heads
        ↓
ReLU
Dropout (0.3)
        ↓
GATConv
128 hidden units × 1 attention head
        ↓
ReLU
        ↓
Global mean pooling
        ↓
Linear classifier
```

Training settings:

```text
Optimizer: Adam
Learning rate: 0.001
Weight decay: 0.0001
Batch size: 8
Maximum epochs: 100
Early-stopping patience: 15
```

The best model is selected using **validation Macro F1**, with validation accuracy used as a tie-breaker.

Run using:

```bash
python model3/model3_gat_residue.py
```

---

# 5.5 Chain-level Graph Representation

A second structural representation models the capsid at the protein-chain level.

### Nodes

Each node represents a protein chain in the structural model.

### Node features

The **same 1,280-dimensional representative-chain ESM-2 embedding** is assigned to every chain node within a graph.

Therefore, differences between chain nodes are not represented by different sequence embeddings. Instead, the graph model can exploit the organization and connectivity of chains.

### Edges

An edge is created between two chains when at least one pair of Cα atoms, one from each chain, has a distance:

```text
< 8 Å
```

Edges are stored in both directions.

Only the first structural model is used.

At least two chains containing Cα coordinates are required to construct an inter-chain graph.

---

## 5.6 Chain-level Dataset

Chain-level graphs could be constructed for **358 entries**.

The reduced sample size reflects the structural requirements of the chain-level representation, including the availability of a compatible PDB structure and the requirement for at least two chains containing Cα coordinates.

The original sequence-level split assignment is retained for entries included in the chain-level subset.

| Split | Graphs |
|---|---:|
| Training | 257 |
| Validation | 50 |
| Test | 51 |
| **Total** | **358** |

The resulting chain-level graph dataset is stored in:

```text
data/graph_dataset.json
```

Construct the dataset using:

```bash
python model3/build_graph_dataset_chain.py
```

---

## 5.7 Chain-level GAT

The chain-level GAT uses:

```text
1280-dimensional node features
        ↓
GATConv
256 hidden units × 4 attention heads
        ↓
ReLU
Dropout (0.3)
        ↓
GATConv
256 hidden units × 1 attention head
        ↓
ReLU
        ↓
Global mean pooling
        ↓
Linear classifier
```

Training settings:

```text
Optimizer: Adam
Learning rate: 0.001
Weight decay: 0.0001
Batch size: 16
Maximum epochs: 100
Early-stopping patience: 15
```

Model selection is based primarily on **validation Macro F1**, with validation accuracy used as a tie-breaker.

Run using:

```bash
python model3/model3_chain.py
```

---

# 5.8 Non-graph XGBoost Baseline

A non-graph XGBoost classifier is evaluated using the same **358-entry chain-level subset**.

The baseline uses only the:

```text
1,280-dimensional representative-chain ESM-2 embedding
```

and does not use:

- graph connectivity
- chain count
- edge features
- other structural graph information

Because the XGBoost model and chain-level GAT use the same subset and split assignments, this comparison evaluates whether the chain-level graph representation provides predictive information beyond the representative sequence embedding.

Run using:

```bash
python model3/model3_xgboost.py
```

---

# 6. Running the Complete Pipeline

All commands should be run from the repository root.

## Step 1 — Retrieve VIPERdb data

```bash
python data_processing/download_viperdb.py
```

## Step 2 — Retrieve resolution and apply structural-quality filtering

```bash
python data_processing/add_resolution.py
```

## Step 3 — Verify entries

```bash
python data_processing/verify_entries.py
```

## Step 4 — Download structures

```bash
python data_processing/download_all_pdb.py
```

## Step 5 — Extract protein sequences

```bash
python data_processing/extract_sequences.py
```

## Step 6 — Reduce sequence redundancy with CD-HIT

```bash
bash data_processing/run_cdhit.sh
```

## Step 7 — Create the train/validation/test split

```bash
python data_processing/split_dataset.py
```

## Step 8 — Generate sequence-level ESM-2 embeddings

```bash
python data_processing/run_esm_embedding.py
```

## Step 9 — Construct the final feature matrix

```bash
python data_processing/build_feature_matrix.py
```

---

# 7. Running the Models

## Model 1

MLP:

```bash
python model1/model1_mlp.py
```

XGBoost baseline:

```bash
python model1/model1_xgboost.py
```

## Model 2

```bash
python model2/model2_regression.py
```

## Model 3 — Residue-level

```bash
python model3/extract_esm_residue.py
python model3/build_graph_dataset_residue.py
python model3/model3_gat_residue.py
```

## Model 3 — Chain-level

```bash
python model3/build_graph_dataset_chain.py
python model3/model3_chain.py
```

## Model 3 — XGBoost baseline

```bash
python model3/model3_xgboost.py
```

---

# 8. Installation

Install Python dependencies using:

```bash
pip install -r requirements.txt
```

CD-HIT is required separately for sequence redundancy reduction.

For example, with Conda:

```bash
conda install -c bioconda cd-hit
```

The neural-network scripts automatically use an available hardware accelerator where supported. CUDA is preferred when available, followed by Apple Metal Performance Shaders (MPS) in scripts supporting macOS acceleration, with CPU used as a fallback.

---

# 9. Large Generated Files

Several generated files are intentionally excluded from the GitHub repository because of their size.

These include:

```text
data/pdb_files_all/
data/esm_residue_embs/
data/graph_dataset_residue.json
```

Approximate local sizes in the final project were:

```text
data/pdb_files_all/               ~4.4 GB
data/esm_residue_embs/            ~974 MB
data/graph_dataset_residue.json   ~5.1 GB
```

These files can be regenerated using the scripts provided in this repository.

Model checkpoint files (`*.pt`) are also generated during training and are not required in the repository because the models can be retrained from the supplied code.

---

# 10. Reproducibility

A fixed random seed of **42** is used where applicable for dataset splitting and model training.

To reduce the risk of information leakage:

- sequence redundancy reduction is performed before dataset splitting
- label encoders for classification are fitted using training labels
- rare-class definitions are derived from the training set
- Model 1 feature standardization is fitted using the training set only
- Model 2 T-number encoding is fitted using the training set only
- validation data are used for model or epoch selection where specified
- the held-out test set is reserved for final evaluation

The fixed dataset split is stored in:

```text
data/dataset_split.json
```

and model configurations and training procedures are defined directly in the corresponding scripts.

---

# 11. Output Files

Final numerical model outputs are stored in:

```text
results/
```

Evaluation figures are stored in:

```text
figures/
```

Examples include the Model 1 confusion matrix and the Model 2 predicted-versus-actual plot.

Generated PyTorch checkpoint files are not required for reproducing the experiments and are excluded from version control.

---

# 12. Summary of Experimental Design

| Model | Task | Input | Structural information |
|---|---|---|---|
| Model 1 MLP | T-number classification | 1280-d mean-pooled ESM-2 embedding | No |
| Model 1 XGBoost | T-number classification | 1280-d mean-pooled ESM-2 embedding | No |
| Model 2 MLP | Outer-diameter regression | 1280-d ESM-2 + one-hot T-number | No explicit structural graph |
| Model 3 Residue GAT | T-number classification | Per-residue 1280-d ESM-2 embeddings | Residue contact graph |
| Model 3 Chain GAT | T-number classification | Representative-chain 1280-d ESM-2 embedding | Chain contact graph |
| Model 3 XGBoost | T-number classification | Representative-chain 1280-d ESM-2 embedding | No |

The overall experimental design therefore compares sequence-only representations with progressively more explicit representations of capsid structural organization.