# gene_mapping.py

import os
import json
import pandas as pd
import anndata as ad
from functools import lru_cache


def _clean_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return ""
    return s


def normalize_gene_symbol(x: str) -> str:
    return _clean_str(x).upper()


def normalize_ensembl_gene_id(x: str) -> str:
    s = _clean_str(x).upper()
    if not s:
        return ""
    if s.startswith("ENSG"):
        return s.split(".")[0]
    return ""


def _to_int_list(values) -> list[int]:
    out = []
    if values is None:
        return out
    for v in values:
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


@lru_cache(maxsize=8)
def load_bmg_feature_list(
    bmg_feature_list_json: str,
    n_col: int,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """
    Returns:
    - feature_name_to_indices: FEATURE_NAME (upper) -> name_to_matrix_col_indices
    - ensembl_to_indices: ENSG... (upper, no version) -> indices from ensembl_gene_ids
    """
    with open(bmg_feature_list_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    feature_name_to_indices: dict[str, list[int]] = {}
    ensembl_to_indices: dict[str, list[int]] = {}

    for rec in records:
        fname = normalize_gene_symbol(rec.get("feature_name", ""))
        name_indices = _to_int_list(rec.get("name_to_matrix_col_indices", []))

        if fname and name_indices:
            bad = [i for i in name_indices if i < 0 or i >= n_col]
            if bad:
                raise ValueError(f"Out-of-range indices for feature_name={fname}: {bad}")
            feature_name_to_indices[fname] = name_indices

        for item in rec.get("ensembl_gene_ids", []) or []:
            ens = normalize_ensembl_gene_id(item.get("id", ""))
            idxs = _to_int_list(item.get("indices", []))
            if not ens or not idxs:
                continue

            bad = [i for i in idxs if i < 0 or i >= n_col]
            if bad:
                raise ValueError(f"Out-of-range indices for ensembl_id={ens}: {bad}")

            ensembl_to_indices[ens] = idxs

    return feature_name_to_indices, ensembl_to_indices


def generate_mapping_table_from_csv(
    cellxgene_var_file: str,
    bmg_feature_list_json: str,
    output_path: str,
    n_col: int,
) -> None:
    """
    Use cellxgene_var_id.csv feature_id (ENSG) to lookup indices, then write soma_joinid as original_index.
    Output CSV columns: index, original_index
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    _, ensembl_to_indices = load_bmg_feature_list(bmg_feature_list_json, n_col=n_col)

    vdf = pd.read_csv(cellxgene_var_file, dtype=str, usecols=["feature_id", "soma_joinid"])
    vdf["feature_id"] = vdf["feature_id"].astype("string").str.strip()
    vdf["soma_joinid"] = vdf["soma_joinid"].astype("string").str.strip()

    records = []
    for _, row in vdf.iterrows():
        fid = _clean_str(row.get("feature_id", ""))
        sj = _clean_str(row.get("soma_joinid", ""))

        if not sj:
            continue

        ens_key = normalize_ensembl_gene_id(fid)
        if not ens_key:
            continue

        idxs = ensembl_to_indices.get(ens_key)
        if not idxs:
            continue

        for idx in idxs:
            records.append({"index": int(idx), "original_index": sj})

    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(output_path, index=False)
    print(f"Mapping table written to {output_path} ({len(out_df)} rows)")


def generate_mapping_table_from_h5ad_single(
    h5ad_path: str,
    bmg_feature_list_json: str,
    output_path: str,
    n_col: int,
) -> None:
    """
    Use adata.var.index (gene symbol) to lookup indices via feature_name.
    Output CSV columns: index, original_index
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    feature_name_to_indices, _ = load_bmg_feature_list(bmg_feature_list_json, n_col=n_col)

    print(f"Loading {h5ad_path} ...")
    adata = ad.read_h5ad(h5ad_path)
    if adata.var.index.duplicated().any():
        raise ValueError(f"Found duplicated gene names in {h5ad_path}")

    records = []
    for gid in adata.var.index.astype(str).tolist():
        key = normalize_gene_symbol(gid)
        if not key:
            continue

        idxs = feature_name_to_indices.get(key)
        if not idxs:
            continue

        for idx in idxs:
            records.append({"index": int(idx), "original_index": key})

    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(output_path, index=False)
    print(f"Mapping table written to {output_path} ({len(out_df)} rows)")
