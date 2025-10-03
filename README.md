# OmniCellTOSG: The First Text–Omic Dataset and Foundation Model for Single-Cell Signaling Graph Modeling and Analysis

<div align="center">
  <img src="./Figures/OmniCell-logo.png" width="40%" alt="OmniCellTOSG" />
</div>

<div align="center" style="line-height: 1;">
  <!-- GitHub -->
  <a href="https://github.com/FuhaiLiAiLab/OmniCellTOSG" target="_blank" style="margin: 2px;">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-OmniCellTOSG%20Code-181717?logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>

  <!-- Hugging Face Dataset -->
  <a href="https://huggingface.co/datasets/FuhaiLiAiLab/OmniCellTOSG_Dataset" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-OmniCellTOSG%20Dataset-ff6f61?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <!-- arXiv -->
  <a href="https://arxiv.org/abs/2504.02148" target="_blank" style="margin: 2px;">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-OmniCellTOSG%20Paper-b31b1b?logo=arxiv&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  
  <!-- License (update if not MIT) -->
  <a href="LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-0a4d92?logo=open-source-initiative&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

---

OmniCellTOSG is, to our knowledge, the first **cell-level Text–Omic dataset** and companion resources for **signaling-graph modeling and analysis** from single-cell data. It integrates quantitative omics with curated textual/semantic annotations to enable graph-language foundation model training, retrieval, and explainable analysis across organs, tissues, cell types, and disease contexts.

<div align="center">
  <img src="./Figures/Figure1.png" alt="OmniCellTOSG Overview" />
</div>

The human body consists of approximately 37 trillion cells, all originating from a single embryonic cell and sharing the same copy of genome. The complex, robust and accurate cell signaling systems, regulated by varying abundance of proteins and their interactions, create diverse cell types with different functions at different organs. The cell signaling systems are evolved and altered by many factors, inlcuding age, sex, diet, environment exposures and diseases. However, the interaction of a multitude of genes and proteins in conjunction introduces complexity in decoding cell signaling systems or patterns in normal development or diseases. The recent advent in the open-source availability of millions of single cell omic data has presented the unique opportunity to integrate multiple layers from the central dogma of molecular biology, for each individual, to unravel how these multi-omic interactions contribute to disease morbidity. Moreover, inspired by the success of foundation models pre-trained on large corpora (e.g., large language models (LLMs) and large vision models (LVMs)), we introduce, to our knowledge, the first dataset of cell-level Text–Omic Signaling Graphs (TOSGs), OmniCellTOSG, together with a unified graph–language foundation model (GLFM), CellTOSG_Foundation (GLFM). In OmniCellTOSG, each TOSG encodes the signaling system of an individual cell or meta-cell and is paired with contextual labels (organ, disease, sex, age, and cell subtype), enabling scalable pretraining and downstream inference over cellular signaling. In sum, we have three major contributions: (i) a Text–Omic Signaling Graph (TOSG) data model that unifies human-interpretable textual priors (functions, locations, pathways, diseases, drugs) with quantitative single-cell measurements, enabling graph interpretation over cell signaling; (ii) a large-scale, training-ready resource built from about 120 millions scRNA-seq cells across diverse tissues and states, packaged in a PyTorch-native format and supported by a rigorous query–loading–balancing pipeline that yields stratified, unbiased cohorts (by tissue, cell type, disease, age, gender, and condition); and (iii) a joint LLM+GNN foundation-model paradigm that fuses language and graph encoders to propagate textual knowledge with omic signals on TOSGs, enabling tasks such as cell-type annotation, classification, signaling inference. Together, OmniCellTOSG provides both the data substrate and modeling framework to accelerate accurate, interpretable decoding of cellular signaling for life sciences and precision medicine.

---

## ⬇️ Download the OmniCellTOSG Dataset
Get the full dataset on HuggingFace: **[OmniCellTOSG_Dataset](https://huggingface.co/datasets/FuhaiLiAiLab/OmniCellTOSG_Dataset)**

---

## 🗂️ Dataset Layout
```
OmniCellTOSG_Dataset/
├─ s_name.csv        # BioMedGraphica_ID ↔ Processed_name (2 columns)
├─ s_desc.csv        # BioMedGraphica_ID ↔ Description (2 columns)
├─ s_bio.csv         # BioMedGraphica_ID ↔ Sequence (2 columns)
│
├─ x_name_emb.npy    # Embeddings for entity names          (483,817 × ?)
├─ x_desc_emb.npy    # Embeddings for entity descriptions   (483,817 × ?)
├─ x_bio_emb.npy     # Embeddings for biological sequences  (483,817 × ?)
│
├─ edge_index.npy            # Full entity graph edges                 (2 × 33,349,084)
├─ internal_edge_index.npy   # Transcript–Protein edges                (2 × 408,299)
├─ ppi_edge_index.npy        # Protein–Protein interaction edges       (2 × 32,940,785)
│
├─ brain/                    # Organ folder
│  ├─ alzheimer's_disease/   # Disease folder
│  │  ├─ brain_sc_alzheimer's_disease_X_partition_0.npy  # scRNA-seq matrix N[0] × 483,817
│  │  ├─ ... 
│  │  ├─ brain_sc_alzheimer's_disease_Y_partition_2.npy  # Labels N[2] × 4
│  │  ├─ ...
│  │  └─ SEA_AD_syn61680896_alzheimer's_disease_Y_partition_26.npy  # Labels N[i] × 4
│  ├─ epilepsy/
│  └─ general/
├─ lung/
├─ kidney/
└─ ...
```

> **Notes**
> - `X_partition_*.npy` files are sparse/stacked partitions of single-cell expression matrices.
> - `Y_partition_*.npy` files contain label matrices with four columns (task-dependent).
> - Embedding array second dimension depends on the encoder used (hence `?`).

---

## ⚙️ Dataset Loader Package Installation
```bash
pip install git+https://github.com/FuhaiLiAiLab/OmniCellTOSG
```

---

## 🐍 Loading Data in Python

```python
import CellTOSGDataset

# Core arguments
x, y, edge_index, internal_edge_index, ppi_edge_index, s_name, s_desc, s_bio = CellTOSGDataset(
    root="</path/to/data>",                                   # Dataset root directory
    categories=["get_organ", "get_disease", "get_organ_disease"],  # Retrieval granularity
    name="brain",  # or "AD" or "brain-AD"                    # Subset name
    label_type="ct",  # or "og" or "ds" or "status"           # Target label type
    seed=2025,
    ratio=0.01,                                               # Sampling ratio for quick experiments
    train_text=False,                                         # If True → return raw text (s_name, s_desc)
    train_bio=False,                                          # If True → return raw sequences (s_bio)
    shuffle=True                                              # Shuffle rows when sampling/merging
)

# Text/sequence payload control
if not train_text:
    x_name_emb, x_desc_emb = x  # precomputed text embeddings
else:
    s_name, s_desc = (s_name, s_desc)  # raw text fields

if not train_bio:
    x_bio_emb = x  # precomputed sequence embeddings
else:
    s_bio = s_bio  # raw biological sequences
```

### 🔧 Parameters
- **root** *(str)*: Filesystem path to the dataset root folder.
- **categories** *(List[str])*: Retrieval scope. Options:
  - `"get_organ"`: choose by organ (e.g., `"brain"`)
  - `"get_disease"`: choose by disease (e.g., `"AD"`)
  - `"get_organ_disease"`: choose by organ–disease pair (e.g., `"brain-AD"`)
- **name** *(str)*: The concrete subset identifier; examples: `"brain"`, `"AD"`, `"brain-AD"`.
- **label_type** *(str)*: Target labels. Options:
  - `"ct"`: cell type
  - `"og"`: organ
  - `"ds"`: disease
  - `"status"`: disease status (e.g., control vs. case)
- **seed** *(int)*: Random seed for sampling/shuffling.
- **ratio** *(float)*: Sampling rate (0–1) for lightweight runs.
- **train_text** *(bool)*: If `True`, return raw text sources (`s_name`, `s_desc`) instead of embeddings.
- **train_bio** *(bool)*: If `True`, return raw sequences (`s_bio`) instead of embeddings.
- **shuffle** *(bool)*: Shuffle examples during sampling or composition.

> **Returns** (typical): 
> - `x, y`: features and labels for the selected split
> - `edge_index`, `internal_edge_index`, `ppi_edge_index`: graph connectivity
> - `s_name`, `s_desc`, `s_bio`: optional raw fields depending on flags

---

## 🌐 Pretrain the Graph Foundation Model
### 🔗 Edge Prediction (Topology Modeling)
Learn topological patterns and interaction mechanisms:
```bash
python pretrain_celltosg.py
```

> Tip: Provide `--config` YAMLs to control backbone, optimizer, scheduler, batch sizes, and checkpoint paths if your repo supports them.

---

## 🚀 Downstream Tasks
### 1) Disease Status Classification
Specify the downstream task and data-loading parameters using the **`CellTOSG_Loader`** package (as invoked in **`train.py`**). Then configure the model hyperparameters to tune performance for your experiment.

```bash
# Alzheimer's Disease (Take AD as an example)
python train.py \
  --downstream_task disease \
  --label_column disease \
  --tissue_general brain \
  --disease_name "Alzheimer's Disease" \
  --sample_ratio 0.1 \
  --train_base_layer gat \
  --train_lr 0.0005 \
  --train_batch_size 3 \
  --train_test_random_seed 42 \
  --dataset_output_dir ./Output/data_ad_disease_0.1
```



### 2) Gender Classification
```bash
# Alzheimer's Disease (Take AD as an example)
python train.py \
  --downstream_task gender \
  --label_column gender \
  --tissue_general brain \
  --disease_name "Alzheimer's Disease" \
  --sample_ratio 0.1 \
  --train_base_layer gat \
  --train_lr 0.0005 \
  --train_batch_size 2 \
  --train_test_random_seed 42 \
  --dataset_output_dir ./Output/data_ad_gender_0.1

```


### 3) Cell Type Annotation
```bash
# Lung (LUAD) (Take LUAD as an example)
python train.py \
  --downstream_task cell_type \
  --label_column cell_type \
  --tissue_general "lung" \
  --disease_name "Lung Adenocarcinoma" \
  --sample_ratio 0.2 \
  --train_base_layer gat \
  --train_lr 0.0001 \
  --train_batch_size 3 \
  --train_test_random_seed 42 \
  --dataset_output_dir ./Output/data_luad_celltype_0.2
```


### 4) Signaling Inference
```bash
python analysis.py
```

## 📜 Citation
If you use this dataset or codebase, please cite:

```bibtex
@article{zhang2025omnicelltosg,
  title     = {OmniCellTOSG: The First Cell Text-Omic Signaling Graphs Dataset for Joint LLM and GNN Modeling},
  author    = {Zhang, Heming and Xu, Tim and Cao, Dekang and Liang, Shunning and Schimmelpfennig, Lars and Kaster, Levi and Huang, Di and Cruchaga, Carlos and Li, Guangfu and Province, Michael and others},
  journal   = {arXiv preprint arXiv:2504.02148},
  year      = {2025}
}
```

---

## 📋 License and Contributions
This project is licensed under the **[MIT License](LICENSE)**, which permits reuse, modification, and distribution for both commercial and non-commercial purposes, provided that the original license is included with any copies of the code.

Contributions are welcome! Please open an issue or submit a pull request following the repository’s contribution guidelines.

