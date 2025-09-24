import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

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

def dataset_correction(downstream_task, expression_matrix, labels, output_dir):

    N_TRANSCRIPT_COLS = 412_039

    # load
    X_full = expression_matrix
    meta   = labels

    # use only transcript features for ComBat; protein tail will be stitched back later
    X = X_full[:, :N_TRANSCRIPT_COLS]

    adata = ad.AnnData(X)
    adata.obs = meta
    adata.var_names = [f"f{i}" for i in range(adata.n_vars)]

    # keep original dataset_id for audit
    adata.obs['dataset_id_raw'] = (
        adata.obs['dataset_id'].astype(str).str.strip()
            .replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'NA': np.nan, 'Unknown': np.nan, "Nan": np.nan, "unknown": np.nan, "NaN": np.nan})
    )

    # build a composite key from 4 columns for assigning missing ids
    batch_cols = ["source", "suspension_type", "tissue_general", "tissue"]
    for c in batch_cols:
        adata.obs[c] = (
            adata.obs[c].astype(str).str.strip()
                .replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'NA': np.nan, 'Unknown': np.nan, "Nan": np.nan, "unknown": np.nan, "NaN": np.nan})
                .fillna('unknown')
        )
    key = adata.obs[batch_cols].agg('|'.join, axis=1)

    # assign new batch_n to missing dataset_id rows
    adata.obs['dataset_id_filled'] = adata.obs['dataset_id_raw'].copy()
    missing = adata.obs['dataset_id_filled'].isna()
    if missing.any():
        uniq_keys = pd.Index(key[missing].unique()).sort_values()
        new_map = {k: f"batch_{i+1:03d}" for i, k in enumerate(uniq_keys)}
        adata.obs.loc[missing, 'dataset_id_filled'] = key[missing].map(new_map)

    # use filled dataset ids directly as batch labels
    adata.obs['dataset_id_filled'] = adata.obs['dataset_id_filled'].astype('category')

    # choose covariate
    if downstream_task.lower() == "disease":
        covariate = "disease_BMG_name"
    else:
        covariate = None

    # ComBat settings
    group_key = "CMT_name"   # stratify by coarse cell type
    MIN_GROUP = 20           # minimal cells per group to run ComBat
    MIN_PER_BATCH = 3        # minimal cells per batch within group

    # clean covariate column if present
    if covariate is not None and covariate in adata.obs.columns:
        adata.obs[covariate] = adata.obs[covariate].astype(str).fillna("unknown")
    else:
        print(f"[INFO] covariate '{covariate}' not found or not set. Run without covariate.")

    adata.layers["X_before_combat"] = adata.X.copy()

    failed_groups, skipped_groups, used_cov_in_groups = [], [], []

    for g in adata.obs[group_key].astype(str).unique():
        idx = (adata.obs[group_key].astype(str) == g)
        n_cells = int(idx.sum())
        print(f"\n[GROUP] {group_key}={g} | n_cells={n_cells}")

        if n_cells < MIN_GROUP:
            print("  -> skipped (too few cells)")
            skipped_groups.append((g, f"too few cells ({n_cells})"))
            continue

        cols = ['dataset_id_filled'] + ([covariate] if covariate and (covariate in adata.obs.columns) else [])
        sub_obs = adata.obs.loc[idx, cols].copy()

        batch_counts = sub_obs['dataset_id_filled'].value_counts()
        ok_batches = batch_counts[batch_counts >= MIN_PER_BATCH].index
        print(f"  batches total={len(batch_counts)}, kept={len(ok_batches)}")

        if len(ok_batches) < 2:
            print("  -> skipped (need >=2 valid batches)")
            skipped_groups.append((g, "need >=2 batches with enough cells"))
            continue

        keep_mask = idx.copy()
        keep_mask[idx] = sub_obs['dataset_id_filled'].isin(ok_batches).values
        sub = adata[keep_mask].copy()

        try:
            if covariate and (covariate in sub.obs.columns):
                sc.pp.combat(sub, key='dataset_id_filled', covariates=[covariate])
                used_cov_in_groups.append(g)
                print("  -> ComBat done with covariate")
            else:
                sc.pp.combat(sub, key='dataset_id_filled')
                print("  -> ComBat done without covariate")
            adata.X[keep_mask, :] = sub.X
        except Exception as e:
            try:
                sc.pp.combat(sub, key='dataset_id_filled')
                adata.X[keep_mask, :] = sub.X
                print("  -> fallback ComBat without covariate success")
            except Exception as e2:
                failed_groups.append((g, repr(e2)))
                print(f"  -> failed: {e2}")

    print(f"\n[INFO] ComBat done. groups used covariate: {len(used_cov_in_groups)}")
    if skipped_groups:
        print("[INFO] skipped groups (first 5):", skipped_groups[:5], "..." if len(skipped_groups) > 5 else "")
    if failed_groups:
        print("[WARN] failed groups (first 5):", failed_groups[:5], "..." if len(failed_groups) > 5 else "")

    # stitch back protein tail and save full matrix (same order as input)
    X_prot_tail = X_full[:, N_TRANSCRIPT_COLS:]
    X_full_corrected = np.concatenate(
        [adata.X.astype(np.float32), X_prot_tail.astype(np.float32)], axis=1
    )
    # np.save(f"expression_matrix_combat.npy", X_full_corrected)

    if output_dir:
        # embedding for visualization
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=50)
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
        sc.tl.umap(adata)

        # plots
        os.makedirs(output_dir, exist_ok=True)

        # Dataset ID (ComBat)
        sc.pl.umap(adata, color=["dataset_id_filled"], title=["ComBat Dataset ID"], show=False)
        plt.savefig(os.path.join(output_dir, "umap_dataset_id_filled.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Original Dataset ID
        sc.pl.umap(adata, color=["dataset_id_raw"], title=["Original Dataset ID"], show=False)
        plt.savefig(os.path.join(output_dir, "umap_dataset_id_raw.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Cell type
        sc.pl.umap(adata, color=["CMT_name"], title=["Cell Type"], show=False)
        plt.savefig(os.path.join(output_dir, "umap_cell_type.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Covariate
        if covariate and (covariate in adata.obs.columns):
            sc.pl.umap(adata, color=[covariate], title=[f"{covariate} (ComBat space)"], show=False)
            plt.savefig(os.path.join(output_dir, f"umap_{covariate}.png"), dpi=300, bbox_inches="tight")
            plt.close()
        else:
            print("[INFO] covariate column missing; skip covariate-colored UMAP.")

    return X_full_corrected
