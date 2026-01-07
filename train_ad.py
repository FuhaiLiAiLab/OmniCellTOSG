from CellTOSG_Loader import CellTOSGDataLoader
import numpy as np
import pandas as pd
import os
from datetime import datetime

data_root = "/storage1/fs1/fuhai.li/Active/tianqi.x/OmniCellTOSG/CellTOSG_dataset_v2"
output_dir = f"./Data/train_dcm_disease_0.1"


dataset = CellTOSGDataLoader(
    root=data_root,
    conditions={
        "tissue_general": "small intestine",
        "disease": "Crohn disease",
    },
    task="disease", # "disease" or "gender" or "cell_type"
    label_column="disease", # "disease" or "gender" or "cell_type"
    sample_ratio=0.1,
    sample_size=None,
    shuffle=True,
    stratified_balancing=True,
    extract_mode="train",
    random_state=2025,
    train_text=False,
    train_bio=False,
    correction_method=None,
    output_dir=output_dir,
)


X = dataset.data            # dict: {"train": train_x, "test": test_x}
Y = dataset.labels          # dict: {"train": y_train, "test": y_test}  or df if no label_column
metadata = dataset.metadata # dict: {"train": train_df, "test": test_df}

print(f"output_dir: {output_dir}")

print("X type:", type(X))
print("X keys:", list(X.keys()) if hasattr(X, "keys") else None)
print("X['train'] type:", type(X["train"]))
print("X['test']  type:", type(X["test"]))
print("X_train shape:", getattr(X["train"], "shape", None))
print("X_test  shape:", getattr(X["test"], "shape", None))

try:
    print("X_train[:2, :10]:\n", X["train"][:2, :10])
except Exception as e:
    print("X_train slicing failed:", repr(e))

try:
    print("X_test[:2, :10]:\n", X["test"][:2, :10])
except Exception as e:
    print("X_test slicing failed:", repr(e))

print("Y type:", type(Y))
if hasattr(Y, "keys"):
    print("Y keys:", list(Y.keys()))
    print("Y['train'] type:", type(Y["train"]))
    print("Y['test']  type:", type(Y["test"]))
    print("Y_train shape:", getattr(Y["train"], "shape", None))
    print("Y_test  shape:", getattr(Y["test"], "shape", None))

    try:
        print("Y_train[:20]:", Y["train"][:20])
    except Exception as e:
        print("Y_train slicing failed:", repr(e))

    try:
        print("Y_test[:20]:", Y["test"][:20])
    except Exception as e:
        print("Y_test slicing failed:", repr(e))
else:
    print("Y is not a dict. Y shape:", getattr(Y, "shape", None))
    try:
        print("Y[:20]:", Y[:20])
    except Exception as e:
        print("Y slicing failed:", repr(e))

print("metadata type:", type(metadata))
print("metadata keys:", list(metadata.keys()) if hasattr(metadata, "keys") else None)

try:
    print("metadata['train'] head:\n", metadata["train"].head())
except Exception as e:
    print("metadata['train'].head() failed:", repr(e))

try:
    print("metadata['test'] head:\n", metadata["test"].head())
except Exception as e:
    print("metadata['test'].head() failed:", repr(e))