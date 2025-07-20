from CellTOSG_Loader import CellTOSGDataLoader
import numpy as np
import pandas as pd
import os

# disease_name = "Alzheimer's Disease"  # Change this to your desired disease name
data_root = "../OmniCellTOSG/dataset_outputs"

dataset = CellTOSGDataLoader(
    root=data_root,
    conditions={
        # "tissue_general": "brain",
        # "tissue": "Cerebral cortex",
        # "cell_type": "glutamatergic neuron",
        # "disease": disease_name,
        # "gender": "female"
    },
    downstream_task="disease", # One of {"disease", "gender", "cell_type"}.
    label_column="disease", # One of {"disease", "gender", "cell_type"}.
    sample_ratio=0.01,
    sample_size=None,
    balanced=False,
    shuffle=True,
    random_state=2025,
    train_text=False,
    train_bio=False,
    output_dir="./Output/data_pretrain_001"  # Change this to your desired output directory
)

X, Y, metadata = dataset.data, dataset.labels, dataset.metadata

# s_name = dataset.s_name
# s_desc = dataset.s_desc
# # s_bio = dataset.s_bio

# x_bio_emb = dataset.x_bio_emb

# print(f"s_name shape: {s_name.shape}")
# print(f"s_desc shape: {s_desc.shape}")

# print(f"x_bio_emb shape: {x_bio_emb.shape}")

print(metadata)
