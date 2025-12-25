import os
import pandas as pd
import numpy as np
from typing import List, Tuple
from .extract import extract_for_inference, extract_for_training
from .data_loader import load_expression_by_metadata, dataset_correction

def sample_matched_by_keys(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    match_keys: List[str],
    stage_col: str = "development_stage_category",
    max_stage_offset: int = 2,
    upsample: bool = True,
    random_state: int = 2025,
):
    """
    Return two values:
      1) matched_target_df – rows from target_df picked to mirror reference_df.
      2) matched_keys_set  – the keys (tuple over match_keys) that were successfully matched.

    - Use CMT_id for matching (fallback to CMT_name if not available).
    - If sex or stage is unknown/empty, loosen matching: match only by cell (CMT_id or CMT_name).
    - Stage back-off is no longer dependent on the position assumption, instead uses stage_col index.
    """

    # ---- Stage definitions and unknown checker ----
    STAGES = [
        "80 and over", "aged", "middle aged", "adult", "young adult", "adolescent",
        "child", "preschool child", "infant", "newborn", "fetal", "embryonic", "unknown",
    ]
    stage_order = {s: i for i, s in enumerate(STAGES)}

    def is_unknown(val) -> bool:
        """Check if a value is unknown or empty."""
        if pd.isna(val):
            return True
        s = str(val).strip().lower()
        return s in {"unknown", "n/a", "na", "none", ""}

    # ---- Validate required columns ----
    if stage_col not in reference_df.columns or stage_col not in target_df.columns:
        raise KeyError(f"stage_col '{stage_col}' must be present in both DataFrames.")

    # Keep only rows where stage_col is not NA (unknown string is still valid)
    reference_df = reference_df[reference_df[stage_col].notna()].copy()
    target_df = target_df[target_df[stage_col].notna()].copy()

    # ---- Locate column indices for keys ----
    try:
        stage_idx = match_keys.index(stage_col)
    except ValueError:
        raise ValueError(f"stage_col '{stage_col}' must appear in match_keys. Got {match_keys}.")

    # Determine cell key: prefer CMT_id, fallback to CMT_name
    if (
        "CMT_id" in reference_df.columns and "CMT_id" in target_df.columns
        and "CMT_id" in match_keys
    ):
        cell_key = "CMT_id"
        cell_idx = match_keys.index("CMT_id")
    elif (
        "CMT_name" in reference_df.columns and "CMT_name" in target_df.columns
        and "CMT_name" in match_keys
    ):
        cell_key = "CMT_name"
        cell_idx = match_keys.index("CMT_name")
    else:
        raise ValueError(
            "Cannot find cell key in both DataFrames and match_keys. "
            "Make sure 'CMT_id' (preferred) or 'CMT_name' is present and included in match_keys."
        )

    # Find sex key if available
    sex_key = None
    sex_idx = None
    for cand in ("sex_normalized", "sex", "gender"):
        if (
            cand in match_keys
            and cand in reference_df.columns
            and cand in target_df.columns
        ):
            sex_key = cand
            sex_idx = match_keys.index(cand)
            break

    # ---- Group by match_keys ----
    ref_groups = reference_df.groupby(match_keys, dropna=False)
    tgt_groups = target_df.groupby(match_keys, dropna=False)

    matched_parts: list[pd.DataFrame] = []
    matched_keys: set = set()

    for key, ref_grp in ref_groups:
        needed = len(ref_grp)
        collected, used_idx = [], set()

        # Determine if loose matching is needed (unknown sex or stage)
        sex_val = key[sex_idx] if sex_idx is not None else None
        stage_val = key[stage_idx]
        need_loose = (sex_idx is not None and is_unknown(sex_val)) or is_unknown(stage_val)

        if need_loose:
            # Loosen matching: match only by cell
            cell_val = key[cell_idx]
            candidates = target_df[target_df[cell_key] == cell_val]

            if len(candidates) > 0:
                take = min(len(candidates), needed)
                if take > 0:
                    sampled = candidates.sample(n=take, replace=False, random_state=random_state)
                    collected.append(sampled)
                    used_idx.update(sampled.index)
                    needed -= take

            # Upsample if still needed
            if needed > 0 and upsample and collected:
                pool = pd.concat(collected, ignore_index=False)
                upsampled = pool.sample(n=needed, replace=True, random_state=random_state)
                collected.append(upsampled)
                needed = 0

        else:
            # ---- Strict matching ----
            # 1) Same stage
            if key in tgt_groups.groups:
                cand = tgt_groups.get_group(key)
                take = min(len(cand), needed)
                if take > 0:
                    sampled = cand.sample(n=take, replace=False, random_state=random_state)
                    collected.append(sampled)
                    used_idx.update(sampled.index)
                    needed -= take

            # 2) Back-off stage: try younger stages
            if not is_unknown(stage_val):
                if stage_val in stage_order:
                    s_idx = stage_order[stage_val]
                    offset = 1
                    while needed > 0 and offset <= max_stage_offset:
                        alt_idx = s_idx + offset
                        if alt_idx >= len(STAGES):
                            break
                        alt_stage = STAGES[alt_idx]
                        alt_key_list = list(key)
                        alt_key_list[stage_idx] = alt_stage
                        alt_key = tuple(alt_key_list)

                        if alt_key in tgt_groups.groups:
                            cand = tgt_groups.get_group(alt_key)
                            cand = cand[~cand.index.isin(used_idx)]
                            take = min(len(cand), needed)
                            if take > 0:
                                sampled = cand.sample(n=take, replace=False, random_state=random_state)
                                collected.append(sampled)
                                used_idx.update(sampled.index)
                                needed -= take

                        offset += 1

            # 3) Upsample if still needed
            if needed > 0 and upsample and collected:
                pool = pd.concat(collected, ignore_index=False)
                upsampled = pool.sample(n=needed, replace=True, random_state=random_state)
                collected.append(upsampled)
                needed = 0

        if collected:
            matched_parts.append(pd.concat(collected, ignore_index=False))
            matched_keys.add(key)

    out_df = pd.concat(matched_parts, ignore_index=True) if matched_parts else pd.DataFrame()
    return out_df, matched_keys

class CellTOSGSubsetBuilder:
    FIELD_ALIAS = {
        "cell_type": "CMT_name",
        "disease": "disease_BMG_name",
        "development_stage": "development_stage_category",
        "gender": "sex_normalized"
    }

    TASK_CONFIG = {
        "pretrain": {},
        "disease": {
            "balance_field": "disease_BMG_name",
            "balance_value": "normal",
            "match_keys": ["CMT_id", "sex_normalized", "development_stage_category"]
        },
        "gender": {
            "balance_field": "sex_normalized",
            "balance_value": "male",
            "match_keys": ["CMT_id", "development_stage_category"]
        },
        "cell_type": {},

    }

    def __init__(self, root):
        self.root = root
        self.bmg_gene_index_csv = os.path.join(root, "bmg_gene_index.csv")
        self.metadata_path = os.path.join(root, "cell_metadata_with_mappings.parquet")
        self.matrix_root = os.path.join(root, "expression_matrix")
        self.df_all = pd.read_parquet(self.metadata_path)
        self.last_query_result = None
        self.last_query_conditions = None

    def _resolve_query_fields(self, query_dict):
        resolved = {}
        for k, v in query_dict.items():
            if v is None:
                continue # Skip None values
            resolved_key = self.FIELD_ALIAS.get(k, k)
            resolved[resolved_key] = v
        return resolved

    def available_conditions(
        self,
        include_fields: List[str] | None = None,
        max_uniques: int | None = None,
    ) -> dict:
        df = self.df_all

        default_fields = [
            "source",
            "suspension_type",
            "tissue_general",
            "tissue",
            "CMT_name",
            "disease_BMG_name",
            "development_stage_category",
            "sex_normalized",
        ]
        fields = include_fields or default_fields
        fields = [f for f in fields if f in df.columns]

        uniques = {}
        for f in fields:
            vals = df[f].dropna().unique().tolist()

            if max_uniques is not None and len(vals) > max_uniques:
                uniques[f] = {
                    "n_unique": int(len(vals)),
                    "n_returned": int(max_uniques),
                    "values": vals[:max_uniques],
                }
            else:
                uniques[f] = {
                    "n_unique": int(len(vals)),
                    "n_returned": int(len(vals)),
                    "values": vals,
                }

        return {
            "status": "NO_SUBSET_RETRIEVED",
            "field_alias": self.FIELD_ALIAS,
            "unique_values": uniques,
            "metadata_rows": int(len(df)),
        }

    def view(self, query_dict):
        query_resolved = self._resolve_query_fields(query_dict)

        mask = pd.Series(True, index=self.df_all.index)

        for k, v in query_resolved.items():
            if k not in self.df_all.columns:
                raise KeyError(f"Column '{k}' not found in metadata.")

            col = self.df_all[k]

            if pd.api.types.is_string_dtype(col):
                col_lower = col.str.lower().fillna("")
                if isinstance(v, (list, tuple, set)):
                    v_lower = [str(i).lower() for i in v]
                    mask &= col_lower.isin(v_lower)
                else:
                    mask &= col_lower == str(v).lower()

            else:
                mask &= col.isin(v) if isinstance(v, (list, tuple, set)) else col == v

        df = self.df_all[mask]
        print(f"Matched {len(df)} samples with given conditions.")

        if df.empty:
            return "No samples matched the query conditions."

        fields_to_report = [
            "source", "suspension_type", "tissue_general", "tissue",
            "CMT_name", "disease_BMG_name", "development_stage_category",
            "sex_normalized", "matrix_file_path"
        ]
        for field in fields_to_report:
            if field not in query_resolved and field in df.columns:
                vc = df[field].value_counts(dropna=False)
                summary = ", ".join([f"{k} ({v})" for k, v in vc.items()])
                print(f"{field}: {summary}")


        disease_key = "disease_BMG_name" if "disease_BMG_name" in query_resolved else None
        if disease_key:
            normal_query = query_resolved.copy()
            normal_query[disease_key] = "normal"
            df_norm = self.df_all.copy()
            for k, v in normal_query.items():
                if isinstance(v, (list, tuple, set)):
                    df_norm = df_norm[df_norm[k].isin(v)]
                else:
                    df_norm = df_norm[df_norm[k] == v]

            print(f"\nMatched {len(df_norm)} 'normal' samples under same conditions:")
            for field in fields_to_report:
                if field not in query_resolved and field in df_norm.columns:
                    vc = df_norm[field].value_counts(dropna=False)
                    summary = ", ".join([f"{k} ({v})" for k, v in vc.items()])
                    print(f"{field} in 'normal': {summary}")

        self.last_query_result = df
        self.last_query_conditions_raw = query_dict
        self.last_query_conditions_resolved = query_resolved
        return df

    def extract(
        self,
        extract_mode=None,  # "inference" | "train"
        task="disease",
        stratified_balancing=False,
        shuffle=False,
        sample_ratio=None,
        sample_size=None,
        random_state=2025,
        correction_method=None, # None / "combat_seq"
        output_dir=None,
    ):

        if self.last_query_result is None:
            return self.available_conditions()

        if task not in self.TASK_CONFIG:
            raise ValueError(f"Unsupported task: {task}")

        if extract_mode not in {"inference", "train"}:
            raise ValueError("Extract mode must be one of {'inference', 'train'}")

        if extract_mode == "inference":
            if task == "pretrain":
                raise ValueError("task='pretrain' is only supported in extract_mode='train'.")
            return extract_for_inference(
                self,
                task=task,
                stratified_balancing=stratified_balancing,
                shuffle=shuffle,
                sample_ratio=sample_ratio,
                sample_size=sample_size,
                random_state=random_state,
                correction_method=correction_method,
                output_dir=output_dir,
            )
    
        elif extract_mode == "train":
            if task == "pretrain" and stratified_balancing:
                print("[Warning] task='pretrain' should not be stratified_balancing; forcing stratified_balancing=False.")
                stratified_balancing = False
            return extract_for_training(
                self,
                task=task,
                stratified_balancing=stratified_balancing,
                shuffle=shuffle,
                sample_ratio=sample_ratio,
                sample_size=sample_size,
                random_state=random_state,
                correction_method=correction_method,
                output_dir=output_dir,
            )
        else:
            raise ValueError("extract_mode must be one of {'inference', 'train'}")