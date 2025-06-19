import os
import numpy as np
import pandas as pd


def load_expression_by_metadata(df_meta, dataset_dir):
    all_expr_parts = []
    sample_indices = []

    for matrix_file in df_meta["matrix_file_path"].unique():
        group = df_meta[df_meta["matrix_file_path"] == matrix_file]
        matrix_path = os.path.join(dataset_dir, matrix_file)
        matrix = np.load(matrix_path, mmap_mode="r")
        print(f"Loaded matrix from {matrix_path} with shape {matrix.shape}")

        row_indices = group["matrix_row_idx"].values
        expr_subset = matrix[row_indices, :]
        all_expr_parts.append(expr_subset)

        sample_indices.extend(group["sample_index"].values)

        del matrix

    expr_all = np.vstack(all_expr_parts)

    sort_order = np.argsort(sample_indices)
    expr_all_sorted = expr_all[sort_order, :]

    return expr_all_sorted