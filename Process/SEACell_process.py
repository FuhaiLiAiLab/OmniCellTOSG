import os
import numpy as np
import pandas as pd
import scanpy as sc
import SEACells
import matplotlib.pyplot as plt
import argparse
from SEACells.cpu import SEACellsCPU
# from SEACells.gpu import SEACellsGPU
from scipy.stats import mode
import scipy.sparse as sp

def write_flag_file(base_dir, subdir, file_name, suffix, reason=None):

    flag_dir = os.path.join(base_dir, subdir, os.path.dirname(file_name))
    os.makedirs(flag_dir, exist_ok=True)
    raw_file_name = os.path.basename(file_name)
    flag_path = os.path.join(flag_dir, os.path.splitext(raw_file_name)[0] + suffix)
    with open(flag_path, "w") as f:
        if reason:
            f.write(str(reason).strip() + "\n")
        else:
            f.write("DONE\n")
    print(f"[INFO] Wrote flag file: {flag_path}")

def create_meta_cells(ad, out_dir, file_name, target_obs, keep_all_obs, donor_obs_keys, input_data_is_log_normalized):

    raw_file_name = os.path.basename(file_name)
    file_str = os.path.splitext(raw_file_name)[0]  # Extract only the file name without path
    print(f"Processing {file_str}...")

    relative_dir = os.path.dirname(file_name)  # Extract the relative directory once
    meta_cells_out_dir = os.path.join(out_dir, "meta_cells", relative_dir)
    os.makedirs(meta_cells_out_dir, exist_ok=True)
    output_file = os.path.join(meta_cells_out_dir, f"{file_str}_SEACells.h5ad")

    if target_obs not in ad.obs.columns:
        print(f"Skipping {file_name}: target_obs '{target_obs}' not found in ad.obs")
        print("N_METACELLS=SKIPPED_MISSING_OBS")
        return ("SKIPPED_MISSING_OBS", 0, "missing target_obs", None)

    if not ad.obs_names.is_unique:
        print("obs_names not unique, fixing...")
        ad.obs_names_make_unique()

    # Minimum cell count check
    CELLS_PER_METACELL = 200
    MIN_METACELLS = 5
    if ad.n_obs < CELLS_PER_METACELL * MIN_METACELLS:
        msg = f"SKIPPED_TOO_FEW_CELLS ({ad.n_obs})"
        print(msg)
        print("N_METACELLS=" + msg)
        return ("SKIPPED_TOO_FEW_CELLS", 0, msg, None)

    # Maximum cell count check
    MAX_CELLS = 150000
    if ad.n_obs > MAX_CELLS:
        msg = f"SKIPPED_TOO_MANY_CELLS ({ad.n_obs})"
        print(msg)
        print("N_METACELLS=" + msg)
        return ("SKIPPED_TOO_MANY_CELLS", 0, msg, None)

    input_data_is_log_normalized = str(input_data_is_log_normalized).lower() in {"true","1","t","yes"}

    # Handle raw based on input_data_is_log_normalized
    def _max_val(mat):
        """Return the maximum value from a dense or sparse matrix as float."""
        return float(mat.max()) if sp.issparse(mat) else float(np.max(mat))

    def _check_if_integer_counts(mat, sample=500_000, tol=1e-6):
        """
        Randomly sample elements and check if they look like integer counts.
        Returns True if >98% of sampled values are within `tol` of the nearest integer.
        """
        arr = mat.data if sp.issparse(mat) else np.asarray(mat).ravel()
        if arr.size == 0:
            return True
        n = min(sample, arr.size)
        idx = np.random.choice(arr.size, n, replace=False)
        frac = np.abs(arr[idx] - np.rint(arr[idx]))
        return (frac < tol).mean() > 0.98

    # Handle raw based on whether input data is log-normalized
    if input_data_is_log_normalized:
        print("Input data is log-normalized. Checking consistency with `ad.raw`...")

        # Check that raw exists
        if ad.raw is None:
            raise ValueError(
                "Input is log-normalized, but `ad.raw` is missing.\n"
                "`create_meta_cells` expects `ad.raw` to store the matrix that will be "
                "aggregated for SEACells (for example, raw or summed counts)."
            )
        raw_max = _max_val(ad.raw.X)
        x_max = _max_val(ad.X)
        print(f"Max value in raw: {raw_max}")
        print(f"Max value in X:   {x_max}")

        # Check if `ad.raw` looks like count data
        if not _check_if_integer_counts(ad.raw.X):
            reason = "RAW_NOT_INTEGER_LIKE (created from ad.X)"
            print(f"WARNING: `ad.raw` does not appear to contain integer-like counts. "
                "Ensure `ad.raw` stores true rawcounts; "
                "it will be used for SEACell aggregation.")
            write_flag_file(out_dir, ".fail", file_name, ".fail", reason)

        # Sanity check for scale consistency
        if x_max >= raw_max:
            print("WARNING: X.max >= raw.max — this suggests `X` may not be log-transformed, "
                "or `ad.raw` may not contain raw counts. Proceed with caution.")

        print("`ad.raw` and `X` check complete. No modifications will be made to `X`.")

    else:
        print("Input data is raw counts. Checking or creating `ad.raw`...")

        if ad.raw is None:
            print("`ad.raw` not found. Creating from `ad.X` (assumed to be raw counts).")
            ad.raw = ad.copy()
            if not _check_if_integer_counts(ad.raw.X):
                reason = "RAW_NOT_INTEGER_LIKE (created from ad.X)"
                print(f"WARNING: `ad.raw` does not appear to contain integer-like counts. "
                    "Ensure `ad.raw` stores true rawcounts; "
                    "it will be used for SEACell aggregation.")
                write_flag_file(out_dir, ".fail", file_name, ".fail", reason)
        else:
            print("`ad.raw` exists.")
            if not _check_if_integer_counts(ad.raw.X):
                reason = "RAW_NOT_INTEGER_LIKE (created from ad.X)"
                print(f"WARNING: `ad.raw` does not appear to contain integer-like counts. "
                    "Ensure `ad.raw` stores true rawcounts; "
                    "it will be used for SEACell aggregation.")
                write_flag_file(out_dir, ".fail", file_name, ".fail", reason)
        
        print("Proceeding with normalization/log1p for visualization only.")
        sc.pp.normalize_total(ad, target_sum=1e4, inplace=True)
        sc.pp.log1p(ad)
        print("Normalization and log1p transformation completed.")

    print(f"Min value in raw: {np.min(ad.raw.X)}")
    print(f"Max value in raw: {np.max(ad.raw.X)}")

    print(f"Min value in X: {np.min(ad.X)}")
    print(f"Max value in X: {np.max(ad.X)}")

    # Perform dimensionality reduction and clustering on X
    # Highly variable genes
    sc.pp.highly_variable_genes(ad, n_top_genes=1500)
    sc.tl.pca(ad, n_comps=50, use_highly_variable=True)

    # neighbors & UMAP on PCA space
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
        plt.close()

    # Metacell analysis using SEACells
    n_SEACells = ad.n_obs // CELLS_PER_METACELL
    print(f"Creating {n_SEACells} metacells from {ad.n_obs} cells")

    try:
        model = SEACellsCPU(ad, build_kernel_on='X_pca', n_SEACells=n_SEACells,
                            n_waypoint_eigs=10, convergence_epsilon=1e-3)
        model.construct_kernel_matrix()
        model.initialize_archetypes()
        model.fit(min_iter=10, max_iter=100)
    except Exception as e:
        msg = f"SEACELLS_FAILED ({str(e)})"
        print(msg)
        print("N_METACELLS=SKIPPED_SEACELL_FAILED")
        return ("FAILED", 0, msg, None)

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

    n_raw_per_meta = ad.obs['SEACell'].value_counts()
    SEACell_ad.obs['n_raw_cells'] = (
        pd.Index(SEACell_ad.obs_names).map(n_raw_per_meta).fillna(0).astype(int)
    )

    # Assign metadata
    if str(keep_all_obs).strip().lower() == "true":
        columns_to_keep = list(ad.obs.columns)
        print(f"[INFO] Keeping ALL obs columns ({len(columns_to_keep)} columns).\nColumns: {columns_to_keep}")
    else:
        default_cols = [
            "dataset_id", "cell_type", "development_stage", "disease",
            "sex", "suspension_type", "tissue", "tissue_general"
        ]
        columns_to_keep = [c for c in default_cols if c in ad.obs.columns]
        print(f"[INFO] Keeping DEFAULT obs columns: {columns_to_keep}")

    donor_col = None
    for key in donor_obs_keys:
        if key in ad.obs.columns:
            donor_col = key
            print(f"[INFO] Found donor key '{donor_col}' in ad.obs.")
            break
    if donor_col is None:
        msg = f"DONOR_KEY_NOT_FOUND ({donor_obs_keys})"
        write_flag_file(out_dir, ".fail", file_name, ".fail", msg)
        raise ValueError(f"None of the donor_obs_keys {donor_obs_keys} found in ad.obs!")

    if donor_col not in columns_to_keep:
        columns_to_keep = [donor_col] + columns_to_keep
        
    sex_col = None
    for c in ["sex"]:
        if c in ad.obs.columns:
            sex_col = c
            break
    if sex_col is None:
        print("[WARNING] No sex column found ('sex'). Sex will not be assigned by donor.")
    else:
        if sex_col not in columns_to_keep:
            columns_to_keep.append(sex_col)

    def _mode_value(s: pd.Series):
        m = s.mode(dropna=True)
        return m.iloc[0] if not m.empty else np.nan

    # Aggregate metadata per metacell
    for col in columns_to_keep:
        if col in ad.obs.columns and col != sex_col:
            agg = ad.obs.groupby("SEACell")[col].agg(_mode_value)
            SEACell_ad.obs[col] = agg.reindex(SEACell_ad.obs_names).values

    if sex_col is not None:
        donor_to_sex = ad.obs.groupby(donor_col)[sex_col].agg(_mode_value)

        if donor_col in SEACell_ad.obs.columns:
            seacell_donor = SEACell_ad.obs[donor_col]
        else:
            seacell_donor = ad.obs.groupby("SEACell")[donor_col].agg(_mode_value).reindex(SEACell_ad.obs_names)

        SEACell_ad.obs[sex_col] = seacell_donor.map(donor_to_sex).values

    SEACell_ad.obs["raw_file"] = raw_file_name

    # Map cell name → index
    obs_index_map = {name: i for i, name in enumerate(ad.obs_names)}
    # Get raw indices per SEACell
    raw_groups = ad.obs.groupby("SEACell").groups  # dict: {metacell → [cell names]}

    SEACell_ad.obs["raw_file_indices"] = [
        ",".join(str(obs_index_map[name]) for name in raw_groups.get(mc, []))
        for mc in SEACell_ad.obs_names
    ]

    # Final obs columns
    final_cols = ["n_raw_cells", "raw_file", "raw_file_indices"] + columns_to_keep
    SEACell_ad.obs = SEACell_ad.obs[final_cols]

    print(f"[INFO] Retained obs columns: {final_cols}")

    # Save processed h5ad
    SEACell_ad.write(output_file)

    print(f"N_METACELLS={SEACell_ad.n_obs}")

    # Compute evaluation metrics
    if target_obs in ad.obs.columns:
        SEACell_purity = SEACells.evaluate.compute_celltype_purity(ad, target_obs)
        SEACell_purity.to_csv(metrics_output_path + '_purity.csv')

    compactness = SEACells.evaluate.compactness(ad, 'X_pca')
    separation = SEACells.evaluate.separation(ad, 'X_pca', nth_nbr=1)

    compactness.to_csv(metrics_output_path + '_compactness.csv')
    separation.to_csv(metrics_output_path + '_separation.csv')

    return ("OK", int(SEACell_ad.n_obs), None, output_file)

def _flag_path(base_dir: str, subdir: str, file_name: str, suffix: str) -> str:
    flag_dir = os.path.join(base_dir, subdir, os.path.dirname(file_name))
    raw_file_name = os.path.basename(file_name)
    return os.path.join(flag_dir, os.path.splitext(raw_file_name)[0] + suffix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Input .h5ad file")
    parser.add_argument("--in_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--target_obs", type=str, required=True, help="Column name in obs for UMAP color and purity calculation")
    parser.add_argument(
        "--keep_all_obs",
        type=str,
        required=True,
        choices=["True", "False"],
        help="If True, keep all obs columns in metacell output; if False, keep only standard metadata"
    )
    parser.add_argument(
        "--donor_obs_keys",
        type=str,
        nargs="+",
        required=True,
        help="One or more candidate obs columns that identify donors"
    )
    parser.add_argument("--input_data_is_log_normalized", type=str, required=True, choices=["True", "False"],
                        help="Indicate if input data is already log-normalized")
    args = parser.parse_args()

    done_path = _flag_path(args.out_dir, ".done", args.file, ".done")
    fail_path = _flag_path(args.out_dir, ".fail", args.file, ".fail")

    if os.path.exists(done_path):
        print(f"[INFO] Skip because done exists: {done_path}")
        print("N_METACELLS=SKIPPED_ALREADY_DONE")
        raise SystemExit(0)

    if os.path.exists(fail_path):
        print(f"[INFO] Skip because fail exists: {fail_path}")
        print("N_METACELLS=SKIPPED_ALREADY_FAILED")
        raise SystemExit(0)

    try:
        # Read input file
        ad = sc.read(os.path.join(args.in_dir, args.file))

        # Run metacell processing ONCE and get status
        status, n_meta, reason, out_path = create_meta_cells(
            ad, args.out_dir, args.file, args.target_obs, args.keep_all_obs, args.donor_obs_keys, args.input_data_is_log_normalized
        )

        if status == "OK":
            # Double-check the output actually exists
            if out_path and os.path.exists(out_path):
                write_flag_file(args.out_dir, ".done", args.file, ".done", f"N_METACELLS={n_meta}")
            else:
                write_flag_file(args.out_dir, ".fail", args.file, ".fail",
                                "FAIL_REASON=OUTPUT_MISSING_AFTER_OK")
        else:
            # failed or skipped; record reason
            write_flag_file(args.out_dir, ".fail", args.file, ".fail",
                            f"FAIL_REASON={reason or status}")

    except Exception as e:
        # Catch any unexpected crash and mark as fail
        write_flag_file(args.out_dir, ".fail", args.file, ".fail",
                        f"FAIL_REASON=UNCAUGHT_EXCEPTION: {repr(e)}")
        raise  # keep traceback in Slurm logs


# Test the script

'''
python SEACell_process.py \
--file ./brain/cerebral_cortex/alzheimer_disease/gse147528_brain_cerebral_cortex_alzheimer_disease_snRNA_entorhinalcortex_annot_partition_0.h5ad \
--in_dir ./braincellatlas_partitioned_raw_data \
--out_dir ./braincellatlas_processed_data_test \
--target_obs "cell_type" \
--keep_all_obs True \
--donor_obs_keys "donor_ID" "pid" \
--input_data_is_log_normalized False

'''