import os
import numpy as np
import pandas as pd
import csv
import scanpy as sc
import anndata as ad
import signal
from contextlib import contextmanager
import matplotlib.pyplot as plt
from typing import Tuple, Optional

@contextmanager
def time_limit(seconds: Optional[float]):
    if seconds is None or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"timeout after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

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

def l1_normalize_log1p(
    matrix: np.ndarray,
    target_sum: float = 1e4,
    eps: float = 1e-12,
    dtype: np.dtype = np.float32,
) -> np.ndarray:

    x = np.asarray(matrix, dtype=np.float64)

    if np.any(x < 0):
        raise ValueError("Input matrices must be non-negative for L1 normalize -> log1p.")

    x_sums = x.sum(axis=1, keepdims=True)

    x_scale = target_sum / np.maximum(x_sums, eps)

    x_norm = np.log1p(x * x_scale)

    return x_norm.astype(dtype, copy=False)


def pad_protein_zeros(matrix: np.ndarray, n_protein_cols: int = 121_419, dtype=None) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {matrix.shape}")
    if n_protein_cols < 0:
        raise ValueError("n_protein_cols must be >= 0")

    if n_protein_cols == 0:
        return matrix

    if dtype is None:
        dtype = matrix.dtype

    zeros = np.zeros((matrix.shape[0], n_protein_cols), dtype=dtype)
    return np.concatenate([matrix, zeros], axis=1)

def combat_seq_correction(
    matrix: np.ndarray,
    meta: pd.DataFrame,
    dataset_id_col: str = "dataset_id",
    dataset_batch_col: str = "dataset_batch",
    batch_id_col: str = "combat_batch",
    covar_cols: Optional[list[str]] = None,
    shrink: bool = False,
    shrink_disp: bool = False,
    gene_subset_n: Optional[int] = None,
    ref_batch: Optional[str] = None,
    na_cov_action: str = "raise",
    timeout_minutes: Optional[float] = None,
) -> np.ndarray:
    try:
        from inmoose.pycombat import pycombat_seq
    except Exception as e:
        raise ImportError(
            "Failed to import inmoose.pycombat.pycombat_seq. Please install inmoose."
        ) from e

    counts = np.asarray(matrix)
    if counts.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape {counts.shape}.")
    if len(meta) != counts.shape[0]:
        raise ValueError("meta rows must match number of samples in matrix.")

    meta = meta.copy()

    batch_source = "unknown"
    if batch_id_col in meta.columns:
        batch = meta[batch_id_col].fillna("NA").astype(str).values
        batch_source = batch_id_col
    else:
        if dataset_id_col not in meta.columns:
            raise KeyError(f"Missing '{dataset_id_col}' in meta.")
        ds = meta[dataset_id_col].fillna("NA").astype(str)
        if dataset_batch_col in meta.columns:
            db = meta[dataset_batch_col].fillna("NA").astype(str)
            batch = (ds + "__" + db).values
            batch_source = f"{dataset_id_col}+{dataset_batch_col}"
        else:
            batch = ds.values
            batch_source = dataset_id_col

    covar_mod = None
    if covar_cols:
        missing = [c for c in covar_cols if c not in meta.columns]
        if missing:
            raise KeyError(f"covar_cols missing in meta: {missing}")
        covar_mod = meta[covar_cols].copy()
        for c in covar_cols:
            covar_mod[c] = covar_mod[c].fillna("NA").astype("category")

    n_samples, n_features = counts.shape
    n_batches = pd.Series(batch).nunique(dropna=False)
    covar_info = covar_cols if covar_cols else None
    print(
        f"[ComBat-Seq] Start: samples={n_samples}, features={n_features}, "
        f"batch_source={batch_source}, n_batches={n_batches}, covar_cols={covar_info}, "
        f"shrink={shrink}, shrink_disp={shrink_disp}, timeout_min={timeout_minutes}"
    )

    if np.any(counts < 0):
        raise ValueError("Counts contain negative values; ComBat-Seq expects non-negative counts.")
    if not np.issubdtype(counts.dtype, np.integer):
        counts = np.rint(counts).astype(np.int64)
    else:
        counts = counts.astype(np.int64, copy=False)

    timeout_seconds = None if timeout_minutes is None else float(timeout_minutes) * 60.0

    try:
        with time_limit(timeout_seconds):
            corrected_gxs = pycombat_seq(
                counts=counts.T,
                batch=batch,
                covar_mod=covar_mod,
                shrink=shrink,
                shrink_disp=shrink_disp,
                gene_subset_n=gene_subset_n,
                ref_batch=ref_batch,
                na_cov_action=na_cov_action,
            )
    except TimeoutError:
        print("[ComBat-Seq] Timeout.")
        return np.asarray(matrix)

    corrected = np.asarray(corrected_gxs).T
    corrected = np.clip(corrected, 0, None)
    if not np.issubdtype(corrected.dtype, np.integer):
        corrected = np.rint(corrected).astype(np.int64)

    print("[ComBat-Seq] Done.")
    return corrected

def _safe_str_col(meta: pd.DataFrame, col: str) -> pd.Series:
    if col not in meta.columns:
        return pd.Series(["NA"] * len(meta), index=meta.index, dtype="object")
    return meta[col].fillna("NA").astype(str)


def build_batch_series(
    meta: pd.DataFrame,
    dataset_id_col: str = "dataset_id",
    fallback_cols: Optional[list[str]] = None,
) -> pd.Series:
    if fallback_cols is None:
        fallback_cols = ["source", "tissue", "suspension_type", "assay"]

    ds = _safe_str_col(meta, dataset_id_col).str.strip()
    ds_lower = ds.str.lower()
    ds_valid = (ds != "") & (~ds_lower.isin({"na", "nan"}))

    parts = [_safe_str_col(meta, c).str.strip() for c in fallback_cols]
    fb = parts[0]
    for p in parts[1:]:
        fb = fb + "__" + p

    return ds.where(ds_valid, fb)

def _normalize_label_col(sub_meta: pd.DataFrame, col: str) -> pd.Series:
    return sub_meta[col].fillna("NA").astype(str).str.strip()


def _eligible_levels(y: pd.Series, min_per_class: int) -> Tuple[int, int, pd.Index]:
    vc = y.value_counts(dropna=False)
    ok_mask = vc >= min_per_class
    ok = int(ok_mask.sum())
    return int(y.nunique(dropna=False)), ok, vc.index[ok_mask]

def _is_strongly_confounded_with_batch(
    sub_meta: pd.DataFrame,
    covar_col: str,
    batch_col: str,
    ok_levels: pd.Index,
) -> Tuple[bool, str]:
    if len(ok_levels) < 2:
        return True, "too_few_ok_levels"

    confounded_levels = []
    for level in ok_levels:
        batches = sub_meta.loc[sub_meta[covar_col] == level, batch_col].astype(str)
        if batches.nunique(dropna=False) < 2:
            confounded_levels.append(str(level))

    if confounded_levels:
        return True, f"levels_in_single_batch={confounded_levels}"

    return False, "pass"

def choose_covar_cols_for_task(
    sub_meta: pd.DataFrame,
    task: str,
    batch_col: str,
    disease_col: str,
    sex_col: str,
    min_per_disease: int,
) -> Tuple[Optional[list[str]], str]:
    if task == "cell_type":
        return None, "no covar (task=cell_type)"

    if task == "sex":
        if sex_col not in sub_meta.columns:
            return None, f"no covar (missing '{sex_col}')"
        sub_meta[sex_col] = _normalize_label_col(sub_meta, sex_col)
        n_cls, ok, ok_levels = _eligible_levels(sub_meta[sex_col], min_per_disease)
        if n_cls < 2:
            return None, f"no covar ({sex_col} n_classes={n_cls})"
        if ok < 2:
            return None, f"no covar ({sex_col} n_classes={n_cls}, ok_classes={ok} < 2)"
        is_conf, reason = _is_strongly_confounded_with_batch(
            sub_meta=sub_meta,
            covar_col=sex_col,
            batch_col=batch_col,
            ok_levels=ok_levels,
        )
        if is_conf:
            return None, f"no covar ({sex_col} confounded_with_batch: {reason})"
        return [sex_col], f"use {sex_col} (n_classes={n_cls}, ok_classes={ok})"

    if task == "disease" or task == "pretrain":
        if disease_col not in sub_meta.columns:
            return None, f"no covar (missing '{disease_col}')"
        sub_meta[disease_col] = _normalize_label_col(sub_meta, disease_col)
        n_cls, ok, ok_levels = _eligible_levels(sub_meta[disease_col], min_per_disease)
        if n_cls < 2:
            return None, f"no covar ({disease_col} n_classes={n_cls})"
        if ok < 2:
            return None, f"no covar ({disease_col} n_classes={n_cls}, ok_classes={ok} < 2)"
        is_conf, reason = _is_strongly_confounded_with_batch(
            sub_meta=sub_meta,
            covar_col=disease_col,
            batch_col=batch_col,
            ok_levels=ok_levels,
        )
        if is_conf:
            return None, f"no covar ({disease_col} confounded_with_batch: {reason})"
        return [disease_col], f"use {disease_col} (n_classes={n_cls}, ok_classes={ok})"

    raise ValueError(f"Unknown task='{task}'. Expected one of: pretrain, disease, sex, cell_type.")


def combat_seq_correction_by_tissue(
    matrix: np.ndarray,
    meta: pd.DataFrame,
    task: str,
    tissue_col: str = "tissue_general",
    disease_col: str = "disease_BMG_name",
    sex_col: str = "sex_normalized",
    dataset_id_col: str = "dataset_id",
    fallback_cols: Optional[list[str]] = None,
    min_batches_per_group: int = 2,
    min_per_disease: int = 5,
    timeout_minutes: Optional[float] = None,
) -> np.ndarray:
    if tissue_col not in meta.columns:
        raise KeyError(f"Missing '{tissue_col}' in meta.")

    corrected = np.asarray(matrix).copy()
    meta = meta.reset_index(drop=True)

    total_groups = meta[tissue_col].nunique(dropna=False)
    print(f"[ComBat-Seq] Grouping by '{tissue_col}', total_groups={total_groups}, task={task}")

    for tissue_value, idx in meta.groupby(tissue_col).groups.items():
        idx = np.array(list(idx), dtype=int)
        sub_meta = meta.iloc[idx].copy()
        sub_mat = corrected[idx]

        sub_meta["combat_batch"] = build_batch_series(
            sub_meta,
            dataset_id_col=dataset_id_col,
            fallback_cols=fallback_cols,
        )

        n_samples = len(sub_meta)
        n_batches_before = sub_meta["combat_batch"].nunique(dropna=False)

        if n_batches_before < min_batches_per_group:
            print(
                f"[ComBat-Seq] Skip tissue='{tissue_value}': samples={n_samples}, "
                f"n_batches={n_batches_before} < {min_batches_per_group}"
            )
            continue

        covar_cols, covar_info = choose_covar_cols_for_task(
            sub_meta=sub_meta,
            task=task,
            batch_col="combat_batch",
            disease_col=disease_col,
            sex_col=sex_col,
            min_per_disease=min_per_disease,
        )

        batch_counts = sub_meta["combat_batch"].value_counts(dropna=False)
        singleton_batches = set(batch_counts[batch_counts < 2].index.tolist())

        if singleton_batches:
            keep_mask = ~sub_meta["combat_batch"].isin(singleton_batches)
            n_drop = int((~keep_mask).sum())
            n_keep = int(keep_mask.sum())
            n_batches_after = sub_meta.loc[keep_mask, "combat_batch"].nunique(dropna=False)

            print(
                f"[ComBat-Seq] tissue='{tissue_value}': drop_singleton_batches_samples={n_drop}, "
                f"kept_samples={n_keep}, batches_before={n_batches_before}, batches_after={n_batches_after}, "
                f"{covar_info}"
            )

            if n_keep == 0 or n_batches_after < min_batches_per_group:
                print(
                    f"[ComBat-Seq] Skip tissue='{tissue_value}' after filtering singletons: "
                    f"kept_samples={n_keep}, n_batches_after={n_batches_after}"
                )
                continue

            sub_meta_keep = sub_meta.loc[keep_mask].copy()
            sub_mat_keep = sub_mat[keep_mask.values]

            covar_cols_keep, covar_info_keep = choose_covar_cols_for_task(
                sub_meta=sub_meta_keep,
                task=task,
                batch_col="combat_batch",
                disease_col=disease_col,
                sex_col=sex_col,
                min_per_disease=min_per_disease,
            )

            try:
                corrected_keep = combat_seq_correction(
                    matrix=sub_mat_keep,
                    meta=sub_meta_keep,
                    batch_id_col="combat_batch",
                    covar_cols=covar_cols_keep,
                    shrink=False,
                    shrink_disp=False,
                    ref_batch=None,
                    na_cov_action="raise",
                    timeout_minutes=timeout_minutes,
                )
            except ValueError as e:
                msg = str(e).lower()
                retryable = (
                    ("confounded" in msg)
                    or ("not full rank" in msg)
                    or ("full rank" in msg)
                    or ("not estimable" in msg)
                    or ("coefficients are not estimable" in msg)
                )
                if retryable and (covar_cols_keep is not None):
                    print(
                        f"[ComBat-Seq] tissue='{tissue_value}': design issue after filtering, "
                        f"retry without covariates. ({covar_info_keep})"
                    )
                    corrected_keep = combat_seq_correction(
                        matrix=sub_mat_keep,
                        meta=sub_meta_keep,
                        batch_id_col="combat_batch",
                        covar_cols=None,
                        shrink=False,
                        shrink_disp=False,
                        ref_batch=None,
                        na_cov_action="raise",
                        timeout_minutes=timeout_minutes,
                    )
                else:
                    raise

            sub_corr = sub_mat.copy()
            sub_corr[keep_mask.values] = corrected_keep
            corrected[idx] = sub_corr
            continue

        print(
            f"[ComBat-Seq] Run tissue='{tissue_value}': samples={n_samples}, "
            f"n_batches={n_batches_before}, {covar_info}"
        )

        try:
            sub_corr = combat_seq_correction(
                matrix=sub_mat,
                meta=sub_meta,
                batch_id_col="combat_batch",
                covar_cols=covar_cols,
                shrink=False,
                shrink_disp=False,
                ref_batch=None,
                na_cov_action="raise",
                timeout_minutes=timeout_minutes,
            )
        except ValueError as e:
            msg = str(e).lower()
            retryable = (
                ("confounded" in msg)
                or ("not full rank" in msg)
                or ("full rank" in msg)
                or ("not estimable" in msg)
                or ("coefficients are not estimable" in msg)
            )
            if retryable and (covar_cols is not None):
                print(f"[ComBat-Seq] tissue='{tissue_value}': design issue, retry without covariates. ({covar_info})")
                sub_corr = combat_seq_correction(
                    matrix=sub_mat,
                    meta=sub_meta,
                    batch_id_col="combat_batch",
                    covar_cols=None,
                    shrink=False,
                    shrink_disp=False,
                    ref_batch=None,
                    na_cov_action="raise",
                    timeout_minutes=timeout_minutes,
                )
            else:
                raise

        corrected[idx] = sub_corr

    print("[ComBat-Seq] Finished all tissue groups.")
    return corrected

def load_bmg_gene_index(csv_path: str) -> list[tuple[str, list[int]]]:
    genes: list[tuple[str, list[int]]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Empty CSV header in bmg_gene_index.csv")
        if "gene_name" not in reader.fieldnames or "indices" not in reader.fieldnames:
            raise ValueError("bmg_gene_index.csv must have columns: gene_name, indices")

        for row in reader:
            gene = (row.get("gene_name") or "").strip()
            idx_str = (row.get("indices") or "").strip()
            if not gene or not idx_str:
                continue

            idx_list: list[int] = []
            for part in idx_str.split(";"):
                part = part.strip()
                if not part:
                    continue
                try:
                    idx_list.append(int(part))
                except ValueError:
                    continue

            if idx_list:
                genes.append((gene, idx_list))

    return genes


def choose_best_column_index(matrix: np.ndarray, indices: list[int]) -> tuple[int, str]:
    if len(indices) == 1:
        chosen = int(indices[0])
        reason = f"Only one candidate index; chose index={chosen}."
        return chosen, reason

    sub = matrix[:, indices]
    n_samples = int(matrix.shape[0])

    nonzero = np.count_nonzero(sub, axis=0)
    best_nz = int(np.max(nonzero))
    tied_pos = np.where(nonzero == best_nz)[0]

    nonzero_fracs = (nonzero / float(n_samples)).astype(float)

    if tied_pos.size == 1:
        chosen = int(indices[int(tied_pos[0])])
        parts = [f"{int(indices[i])}:{nonzero_fracs[i]:.6f}" for i in range(len(indices))]
        reason = (
            f"Chose index={chosen} because it has the highest nonzero fraction "
            f"({(best_nz / float(n_samples)):.6f}). "
            f"Nonzero fractions by index: " + "; ".join(parts)
        )
        return chosen, reason

    if best_nz == 0:
        chosen = int(indices[0])
        idx_list_str = ", ".join(str(int(i)) for i in indices)
        reason = (
            f"All candidates are all-zero columns (nonzero fraction 0.000000). "
            f"Chose index={chosen} by stable rule: pick the first index from [{idx_list_str}]."
        )
        return chosen, reason

    totals = np.sum(sub[:, tied_pos], axis=0, dtype=np.int64)
    best_tied = int(np.argmax(totals))
    chosen = int(indices[int(tied_pos[best_tied])])

    parts_nonzero = [f"{int(indices[i])}:{nonzero_fracs[i]:.6f}" for i in range(len(indices))]
    parts_total = [f"{int(indices[int(tied_pos[j])])}:{int(totals[j])}" for j in range(len(tied_pos))]

    reason = (
        f"Nonzero fraction tied at {(best_nz / float(n_samples)):.6f}. "
        f"Chose index={chosen} because it has the largest total count among tied indices. "
        f"Nonzero fractions by index: " + "; ".join(parts_nonzero) + ". "
        f"Total counts among tied indices: " + "; ".join(parts_total)
    )
    return chosen, reason

def bmg_matrix_to_gene_matrix(
    bmg_matrix: np.ndarray,
    bmg_gene_index_csv: str,
    n_col_expected: int,
) -> Tuple[np.ndarray, list[str], pd.DataFrame]:
    mat = np.asarray(bmg_matrix)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={mat.shape}")

    n_samples, n_cols = mat.shape
    if n_cols != int(n_col_expected):
        raise ValueError(f"Column count mismatch: got {n_cols}, expected {n_col_expected}")

    genes = load_bmg_gene_index(bmg_gene_index_csv)

    gene_names: list[str] = []
    chosen_indices: list[int] = []
    candidate_strs: list[str] = []
    reasons: list[str] = []

    for gene, indices in genes:
        bad = [i for i in indices if i < 0 or i >= n_cols]
        if bad:
            raise ValueError(f"Out-of-range indices for gene={gene}: {bad}")

        chosen, reason = choose_best_column_index(mat, indices)
        gene_names.append(gene)
        chosen_indices.append(int(chosen))
        candidate_strs.append(";".join(str(int(i)) for i in indices))
        reasons.append(reason)

    chosen_idx_arr = np.asarray(chosen_indices, dtype=np.int64)
    gene_matrix = mat[:, chosen_idx_arr]

    choice_df = pd.DataFrame(
        {
            "gene_name": gene_names,
            "chosen_bmg_index": chosen_indices,
            "candidate_bmg_indices": candidate_strs,
            "choice_reason": reasons,
        }
    )
    return gene_matrix, gene_names, choice_df

def build_gene_df_from_bmg_matrix(
    bmg_matrix: np.ndarray,
    meta: pd.DataFrame,
    bmg_gene_index_csv: str,
    n_col_expected: int,
    sample_id_col: str = "sample_index",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if sample_id_col not in meta.columns:
        raise KeyError(f"Missing '{sample_id_col}' in meta.")

    gene_matrix, gene_names, choice_df = bmg_matrix_to_gene_matrix(
        bmg_matrix=bmg_matrix,
        bmg_gene_index_csv=bmg_gene_index_csv,
        n_col_expected=n_col_expected,
    )

    sample_ids = meta[sample_id_col].astype(int).tolist()
    gene_df = pd.DataFrame(gene_matrix, index=sample_ids, columns=gene_names)
    gene_df.index.name = sample_id_col
    return gene_df, choice_df



def build_gene_df(
    final_df: pd.DataFrame,
    expression_matrix: np.ndarray,
    bmg_gene_index_csv: str,
    n_col_expected: int,
    task: str,
    correction_method: Optional[str] = None,
    tissue_col: str = "tissue_general",
    disease_col: str = "disease_BMG_name",
    sex_col: str = "sex_normalized",
    dataset_id_col: str = "dataset_id",
    fallback_cols: Optional[list[str]] = None,
    min_batches_per_group: int = 2,
    min_per_disease: int = 5,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        gene_df_raw: sample x gene (no correction)
        gene_df_corrected: sample x gene (corrected) or None
        final_df_out: metadata aligned with gene_df rows (has sample_index)
        choice_df: gene -> chosen BMG column mapping
    """
    final_df_out = final_df.copy()
    if "sample_index" not in final_df_out.columns:
        final_df_out["sample_index"] = np.arange(len(final_df_out), dtype=int)

    if fallback_cols is None:
        fallback_cols = ["source", "tissue", "suspension_type", "assay"]

    gene_df_raw, choice_df = build_gene_df_from_bmg_matrix(
        bmg_matrix=expression_matrix,
        meta=final_df_out,
        bmg_gene_index_csv=bmg_gene_index_csv,
        n_col_expected=n_col_expected,
        sample_id_col="sample_index",
    )

    method = (str(correction_method).strip().lower() if correction_method is not None else None)

    if method in (None, "", "none"):
        return gene_df_raw, None, final_df_out, choice_df

    if method not in ("combat", "combat_seq", "pycombat_seq"):
        raise ValueError(f"Unsupported correction_method: {correction_method}")

    corrected_counts = combat_seq_correction_by_tissue(
        matrix=gene_df_raw.to_numpy(),
        meta=final_df_out,
        task=task,
        tissue_col=tissue_col,
        disease_col=disease_col,
        sex_col=sex_col,
        dataset_id_col=dataset_id_col,
        fallback_cols=fallback_cols,
        min_batches_per_group=min_batches_per_group,
        min_per_disease=min_per_disease,
        timeout_minutes=60,
    )

    gene_df_corrected = pd.DataFrame(
        corrected_counts,
        index=gene_df_raw.index,
        columns=gene_df_raw.columns,
    )
    gene_df_corrected.index.name = gene_df_raw.index.name

    return gene_df_raw, gene_df_corrected, final_df_out, choice_df