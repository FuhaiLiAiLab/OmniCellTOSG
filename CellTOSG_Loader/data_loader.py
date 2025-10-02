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

def dataset_correction(
    downstream_task,
    expression_matrix,
    labels,
    output_dir,
    MIN_GROUP=20,
    MIN_PER_BATCH=3
):

    N_TRANSCRIPT_COLS = 412_039

    # Load
    X_full = np.array(expression_matrix, copy=True)
    meta   = labels.copy()

    # Transcript block
    X = X_full[:, :N_TRANSCRIPT_COLS].copy()

    adata = ad.AnnData(X)
    adata.obs = meta
    adata.var_names = [f"f{i}" for i in range(adata.n_vars)]

    # Keep original dataset_id
    adata.obs['dataset_id_raw'] = (
        adata.obs['dataset_id'].astype(str).str.strip()
            .replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'NA': np.nan,
                      'Unknown': np.nan, 'unknown': np.nan, 'Nan': np.nan, 'NaN': np.nan})
    )

    # Composite key to fill missing ids
    batch_cols = ["source", "suspension_type", "tissue_general", "tissue"]
    for c in batch_cols:
        adata.obs[c] = (
            adata.obs[c].astype(str).str.strip()
                .replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'NA': np.nan,
                          'Unknown': np.nan, 'unknown': np.nan, 'Nan': np.nan, 'NaN': np.nan})
                .fillna('unknown')
        )
    key = adata.obs[batch_cols].agg('|'.join, axis=1)

    # Assign synthetic batch to missing dataset_id rows
    adata.obs['dataset_id_filled'] = adata.obs['dataset_id_raw'].copy()
    missing = adata.obs['dataset_id_filled'].isna()
    if missing.any():
        uniq_keys = pd.Index(key[missing].unique()).sort_values()
        new_map = {k: f"batch_{i+1:03d}" for i, k in enumerate(uniq_keys)}
        adata.obs.loc[missing, 'dataset_id_filled'] = key[missing].map(new_map)

    # Batch labels to category
    adata.obs['dataset_id_filled'] = adata.obs['dataset_id_filled'].astype('category')

    # Disease covariate
    if downstream_task.lower() == "disease" and ("disease_BMG_name" in adata.obs.columns):
        adata.obs["disease_BMG_name"] = (
            adata.obs["disease_BMG_name"].astype(str).str.strip()
                .replace({'': np.nan, 'nan': np.nan, 'None': np.nan, 'NA': np.nan,
                          'Unknown': np.nan, 'unknown': np.nan, 'Nan': np.nan, 'NaN': np.nan})
                .fillna('unknown')
        )
        disease_cov = "disease_BMG_name"
    else:
        print("[INFO] no disease covariate or not in disease mode; running without it.")
        disease_cov = None

    adata.layers["X_before_combat"] = adata.X.copy()

    failed_groups, skipped_groups = [], []
    used_cov_stage1, ran_stage2 = [], []

    group_key = "CMT_name"

    for g in adata.obs[group_key].astype(str).unique():
        idx = (adata.obs[group_key].astype(str) == g)
        n_cells = int(idx.sum())
        print(f"\n[GROUP] {group_key}={g} | n_cells={n_cells}")

        if n_cells < MIN_GROUP:
            print("  -> skipped (too few cells)")
            skipped_groups.append((g, f"too few cells ({n_cells})"))
            continue

        # Keep batches with enough cells
        cols_needed = ["dataset_id_filled", "suspension_type"] + ([disease_cov] if disease_cov else [])
        sub_obs = adata.obs.loc[idx, cols_needed]
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
        Xw = np.asarray(sub.X)

        # ComBat by dataset_id
        covs_stage1 = [disease_cov] if disease_cov else []

        std = np.nanstd(Xw, axis=0)
        zv_mask = (std == 0) | ~np.isfinite(std)
        work_mask = ~zv_mask
        if work_mask.sum() == 0:
            print("  -> all-zero-variance; set block to 0 and skip ComBat (stage 1)")
            Xw[:, :] = 0.0
        else:
            try:
                sub_work = ad.AnnData(Xw[:, work_mask].copy(), obs=sub.obs.copy())
                if covs_stage1:
                    sc.pp.combat(sub_work, key='dataset_id_filled', covariates=covs_stage1)
                    used_cov_stage1.append(g)
                    print("  -> Stage 1 ComBat (by dataset_id) WITH covariates:", covs_stage1)
                else:
                    sc.pp.combat(sub_work, key='dataset_id_filled')
                    print("  -> Stage 1 ComBat (by dataset_id) WITHOUT covariates")
                Xw[:, work_mask] = sub_work.X
            except Exception as e:
                print(f"  -> Stage 1 fallback (no covariates) due to: {e}")
                try:
                    sub_work = ad.AnnData(Xw[:, work_mask].copy(), obs=sub.obs.copy())
                    sc.pp.combat(sub_work, key='dataset_id_filled')
                    Xw[:, work_mask] = sub_work.X
                except Exception as e2:
                    failed_groups.append((g, f"stage1 {repr(e2)}"))

        Xw[:, zv_mask] = 0.0
        sub.X = Xw

        # ComBat by suspension_type
        if "suspension_type" in sub.obs.columns:
            mod_counts = sub.obs["suspension_type"].value_counts()
            n_modalities = mod_counts.shape[0]
            if n_modalities == 2 and mod_counts.min() >= 3:
                sub2 = sub.copy()
                X2 = np.asarray(sub2.X)

                # Run ComBat by suspension_type on variable columns;
                std2 = np.nanstd(X2, axis=0)
                zv2 = (std2 == 0) | ~np.isfinite(std2)
                work2 = ~zv2
                try:
                    if work2.sum() > 0:
                        sub2_work = ad.AnnData(X2[:, work2].copy(), obs=sub2.obs.copy())
                        covs_stage2 = [disease_cov] if disease_cov else []
                        if covs_stage2:
                            sc.pp.combat(sub2_work, key='suspension_type', covariates=covs_stage2)
                            print("  -> Stage 2 ComBat (by suspension_type) WITH covariates:", covs_stage2)
                        else:
                            sc.pp.combat(sub2_work, key='suspension_type')
                            print("  -> Stage 2 ComBat (by suspension_type) WITHOUT covariates")
                        X2[:, work2] = sub2_work.X
                    X2[:, zv2] = 0.0
                    sub2.X = X2

                    # Write back to sub
                    sub.X[:, :] = sub2.X
                    ran_stage2.append(g)
                except Exception as e:
                    print(f"  -> Stage 2 skipped for group {g} due to error: {e}")
            else:
                print(f"  -> Stage 2 skipped (need 2 suspension_type with >=3 cells each; got {mod_counts.to_dict()})")

        # Cleanup and write back to the main matrix
        bad = ~np.isfinite(sub.X)
        if np.any(bad):
            n_bad = int(bad.sum())
            print(f"  [WARN] subset has {n_bad} NaN/Inf; setting to 0.")
            sub.X[bad] = 0.0

        adata.X[keep_mask, :] = sub.X

    print(f"\n[INFO] Stage 1 used disease covariate in {len(used_cov_stage1)} groups.")
    print(f"[INFO] Stage 2 ran in {len(ran_stage2)} groups.")
    if skipped_groups:
        print("[INFO] skipped groups (first 5):", skipped_groups[:5], "..." if len(skipped_groups) > 5 else "")
    if failed_groups:
        print("[WARN] failed groups (first 5):", failed_groups[:5], "..." if len(failed_groups) > 5 else "")

    Xg = np.asarray(adata.X)
    if np.any(~np.isfinite(Xg)):
        n_bad = int((~np.isfinite(Xg)).sum())
        print(f"[WARN] global NaN/Inf={n_bad}; set to 0.")
        Xg = np.nan_to_num(Xg, nan=0.0, posinf=0.0, neginf=0.0)
    col_std = np.std(Xg, axis=0)
    glob_zv = (col_std == 0) | ~np.isfinite(col_std)
    if glob_zv.any():
        print(f"[INFO] global zero-variance genes: {int(glob_zv.sum())} (set to 0)")
        Xg[:, glob_zv] = 0.0
    adata.X = Xg

    # Stitch back protein tail and return
    X_prot_tail = X_full[:, N_TRANSCRIPT_COLS:]
    X_full_corrected = np.concatenate(
        [adata.X.astype(np.float32), X_prot_tail.astype(np.float32)], axis=1
    )

    if output_dir:
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=50)
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
        sc.tl.umap(adata)

        os.makedirs(output_dir, exist_ok=True)
        sc.pl.umap(adata, color=["dataset_id_filled"], title=["ComBat Dataset ID"], show=False)
        plt.savefig(os.path.join(output_dir, "umap_dataset_id_filled.png"), dpi=300, bbox_inches="tight"); plt.close()
        sc.pl.umap(adata, color=["CMT_name"], title=["Cell Type"], show=False)
        plt.savefig(os.path.join(output_dir, "umap_cell_type.png"), dpi=300, bbox_inches="tight"); plt.close()
        if "suspension_type" in adata.obs.columns:
            sc.pl.umap(adata, color=["suspension_type"], title=["suspension_type (post-ComBat)"], show=False)
            plt.savefig(os.path.join(output_dir, "umap_suspension_type.png"), dpi=300, bbox_inches="tight"); plt.close()
        if disease_cov:
            sc.pl.umap(adata, color=[disease_cov], title=[f"{disease_cov} (post-ComBat)"], show=False)
            plt.savefig(os.path.join(output_dir, f"umap_{disease_cov}.png"), dpi=300, bbox_inches="tight"); plt.close()

    return X_full_corrected