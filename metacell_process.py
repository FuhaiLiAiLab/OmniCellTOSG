import os
import numpy as np
import pandas as pd
import scanpy as sc
import SEACells
import matplotlib.pyplot as plt
import argparse
from SEACells.cpu import SEACellsCPU
from scipy.stats import mode

def create_meta_cells(ad, out_dir, file_name, target_obs, obs_columns, input_data_is_log_normalized):

    file_str = os.path.splitext(os.path.basename(file_name))[0]  # Extract only the file name without path
    print(f"Processing {file_str}...")

    relative_dir = os.path.dirname(file_name)  # Extract the relative directory once
    meta_cells_out_dir = os.path.join(out_dir, "meta_cells", relative_dir)
    os.makedirs(meta_cells_out_dir, exist_ok=True)
    output_file = os.path.join(meta_cells_out_dir, f"{file_str}_SEACells.h5ad")

    # Check if the output file already exists
    if os.path.exists(output_file):
        print(f"Output file '{output_file}' already exists. Skipping processing.")
        return

    # Check if target_obs exists in ad.obs.columns
    if target_obs not in ad.obs.columns:
        print(f"Skipping {file_name}: target_obs '{target_obs}' not found in ad.obs")
        return
    
    output_columns = ["disease", "organ", "substructure", "cell_type"]
    if len(obs_columns) != len(output_columns):
        raise ValueError(f"Error: The number of input columns ({len(obs_columns)}) does not match the expected number ({len(output_columns)}).")
    
    if not ad.obs_names.is_unique:
        print("obs_names not unique, fixing...")
        ad.obs_names_make_unique()

    input_data_is_log_normalized = args.input_data_is_log_normalized.lower() == "true"

    # Handle raw layer based on input_data_is_log_normalized
    if input_data_is_log_normalized:
        print("Input data is log-normalized. Checking consistency with raw layer...")
        if ad.raw is None:
            raise ValueError("Error: Input data is log-normalized but no raw layer is provided.")
        
        raw_max = np.max(ad.raw.X)
        x_max = np.max(ad.X)
        
        print(f"Max value in raw layer: {raw_max}")
        print(f"Max value in X layer: {x_max}")

        if raw_max == x_max:
            raise ValueError("Error: The raw layer and X layer have identical maximum values, indicating potential data issues.")
        print("Raw layer and X layer check passed: values are consistent.")
    else:
        print("Input data is raw counts. Checking raw layer if exists...")
        
        if ad.raw is not None:
            raw_max = np.max(ad.raw.X)
            x_max = np.max(ad.X)

            print(f"Max value in raw layer: {raw_max}")
            print(f"Max value in X layer: {x_max}")

            if raw_max == x_max:
                raise ValueError("Error: The raw layer and X layer have identical maximum values, indicating potential data issues.")
            print("Raw layer and X layer check passed: values are consistent.")
        else:
            print("Raw layer not found. Creating raw layer from X...")
            ad.raw = ad.copy()
            print("Raw layer created successfully.")

        print("Applying normalization and log1p transformation...")
        sc.pp.normalize_total(ad, target_sum=1e4, inplace=True)
        sc.pp.log1p(ad)
        print("Normalization and log1p transformation completed.")

    print(f"Min value in raw: {np.min(ad.raw.X)}")
    print(f"Max value in raw: {np.max(ad.raw.X)}")

    print(f"Min value in X: {np.min(ad.X)}")
    print(f"Max value in X: {np.max(ad.X)}")

    # Perform dimensionality reduction and clustering on X
    sc.pp.highly_variable_genes(ad, n_top_genes=1500)
    sc.tl.pca(ad, n_comps=50, use_highly_variable=True)
    sc.pp.neighbors(ad, use_rep='X_pca')
    sc.tl.umap(ad)

    # metrics output directory
    meta_cell_metrics_out_dir = os.path.join(out_dir, "meta_cell_metrics", relative_dir)
    os.makedirs(meta_cell_metrics_out_dir, exist_ok=True)
    metrics_output_path = os.path.join(meta_cell_metrics_out_dir, file_str)

    # Debugging outputs
    print(f"DEBUG: file_name = {file_name}")
    # print(f"DEBUG: relative_dir = {relative_dir}")
    # print(f"DEBUG: meta_cells_out_dir = {meta_cells_out_dir}")
    # print(f"DEBUG: meta_cell_metrics_out_dir = {meta_cell_metrics_out_dir}")
    # print(f"DEBUG: metrics_output_path = {metrics_output_path}")

    # Save UMAP plot with user-specified target_obs
    if target_obs in ad.obs.columns:
        sc.pl.scatter(ad, basis='umap', color=target_obs, frameon=False)
        plt.savefig(metrics_output_path + "_UMAP.png", dpi=300)

    # Metacell analysis using SEACells
    n_SEACells = int(ad.n_obs / 200)
    print(f'Creating {n_SEACells} metacells from {ad.n_obs} cells')
    model = SEACellsCPU(ad, build_kernel_on='X_pca', n_SEACells=n_SEACells, n_waypoint_eigs=10, convergence_epsilon=1e-3)
    model.construct_kernel_matrix()
    model.initialize_archetypes()
    model.fit(min_iter=10, max_iter=100)

    # Assign metacells
    hard_assignments = model.get_hard_assignments()
    ad.obs['SEACell'] = hard_assignments['SEACell']

    SEACells.plot.plot_2D(ad, key='X_umap', colour_metacells=True, save_as=metrics_output_path+"_meta_cell_UMAP.png")
    
    # Summarize SEACells
    SEACell_ad = SEACells.core.summarize_by_SEACell(ad, SEACells_label='SEACell', summarize_layer='raw')
    

    print(f"Min value in raw: {np.min(ad.raw.X)}")
    print(f"Max value in raw: {np.max(ad.raw.X)}")

    print(f"Min value in X: {np.min(ad.X)}")
    print(f"Max value in X: {np.max(ad.X)}")

    print(f"Min value in SEACell X: {np.min(SEACell_ad.X)}")
    print(f"Max value in SEACell X: {np.max(SEACell_ad.X)}")

    sc.pp.normalize_total(SEACell_ad, target_sum=1e4, inplace=True)
    sc.pp.log1p(SEACell_ad)

    print(f"Min value in SEACell X: {np.min(SEACell_ad.X)}")
    print(f"Max value in SEACell X: {np.max(SEACell_ad.X)}")

    # Assign metadata
    for col in obs_columns:
        if col in ad.obs.columns:
            SEACell_ad.obs[col] = (
                ad.obs.groupby('SEACell')[col]
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            )

    # Step 1: Rename all non-empty columns
    for original_col, output_col in zip(obs_columns, output_columns):
        if original_col.strip() != "":  # Handle space input
            if original_col in SEACell_ad.obs.columns:
                SEACell_ad.obs.rename(columns={original_col: output_col}, inplace=True)
                print(f"Renamed column '{original_col}' to '{output_col}'.")
            else:
                print(f"Warning: Column '{original_col}' not found in SEACell_ad.obs.")

    # Step 2: Handle empty column
    if "organ" not in SEACell_ad.obs.columns:
        SEACell_ad.obs["organ"] = "unknown"
        print("'organ' column missing. Filled with 'unknown'.")

    if "substructure" not in SEACell_ad.obs.columns:
        SEACell_ad.obs["substructure"] = "unknown"
        print("'substructure' column missing. Filled with 'unknown'.")


    # Save processed h5ad
    SEACell_ad.write(output_file)

    # Compute evaluation metrics
    if target_obs in ad.obs.columns:
        SEACell_purity = SEACells.evaluate.compute_celltype_purity(ad, target_obs)
        SEACell_purity.to_csv(metrics_output_path + '_purity.csv')

    compactness = SEACells.evaluate.compactness(ad, 'X_pca')
    separation = SEACells.evaluate.separation(ad, 'X_pca', nth_nbr=1)

    compactness.to_csv(metrics_output_path + '_compactness.csv')
    separation.to_csv(metrics_output_path + '_separation.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Input .h5ad file")
    parser.add_argument("--in_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--target_obs", type=str, required=True, help="Column name in obs for UMAP color and purity calculation")
    parser.add_argument("--obs_columns", type=str, nargs='+', required=True, help="List of obs columns to keep in meta cells")
    parser.add_argument("--input_data_is_log_normalized", type=str, required=True, choices=["True", "False"], 
                        help="Indicate if input data is already log-normalized")
    args = parser.parse_args()

    # Read input file
    ad = sc.read(os.path.join(args.in_dir, args.file))

    # Run metacell processing
    create_meta_cells(ad, args.out_dir, args.file, args.target_obs, args.obs_columns, args.input_data_is_log_normalized)