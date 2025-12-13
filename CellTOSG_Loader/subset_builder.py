import os
import pandas as pd
import numpy as np
from typing import List, Tuple
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
    if (
        "sex_normalized" in match_keys
        and "sex_normalized" in reference_df.columns
        and "sex_normalized" in target_df.columns
    ):
        sex_key = "sex_normalized"
        sex_idx = match_keys.index("sex_normalized")

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
        shuffle=False,
        balanced=False,
        downstream_task="disease",
        output_dir=None,
        sample_ratio=None,
        sample_size=None,
        random_state=2025
    ):
        
        """Extract expression matrix & labels for modelling.

        Parameters
        ----------
        shuffle        Shuffle rows before exporting.
        balanced       Perform class balancing only for tasks that support it.
        downstream_task One of {"disease", "gender", "cell_type"}.
        output_dir     If provided, persist labels + expression to this directory.
        sample_ratio   Fraction (0‑1) of rows to keep.
        sample_size    Exact number of rows to keep.
        random_state   Seed for deterministic sampling.
        """

        if self.last_query_result is None:
            raise ValueError("Please call .view() first to select your subset.")

        if downstream_task not in self.TASK_CONFIG:
            raise ValueError(f"Unsupported downstream_task: {downstream_task}")

        if sample_size is not None and sample_ratio is not None:
            print(
                "[Warning] Both sample_size and sample_ratio provided; "
                "sample_ratio will take precedence."
            )
            sample_size = None  # ignore size in favour of ratio
    
        if downstream_task == "cell_type":
            if balanced:
                print("[Warning] downstream_task='cell_type' should not be balanced，setting balanced=False.")
                balanced = False

            df = self.last_query_result.copy()

            # Drop rows with missing cell type
            df = df[
                df["CMT_name"].notna()
                & (df["CMT_name"].astype(str).str.strip() != "")
                & (~df["CMT_name"].astype(str).str.lower().isin(
                    ["unannoted", "unannotated", "unknown", "miscellaneous", "splatter", "cell"]
                ))
            ].copy()
            # ---- plain sampling (no balancing) ----
            if sample_ratio is not None:
                grouped = df.groupby("CMT_name")
                sampled_parts = []
                for name, group in grouped:
                    n_total = len(group)
                    n_sample = max(int(n_total * sample_ratio), 10)
                    if n_total < n_sample:
                        print(f"[Warning] cell type '{name}' has only {n_total} samples; upsampling to {n_sample}.")
                        sampled = group.sample(n_sample, replace=True, random_state=random_state)
                    else:
                        sampled = group.sample(n=n_sample, replace=False, random_state=random_state)
                        print(f"[Info] Sampled {len(sampled)} from cell type '{name}' with {n_total} total samples.")
                    sampled_parts.append(sampled)
                df = pd.concat(sampled_parts, ignore_index=True)

            elif sample_size is not None:
                take = min(sample_size, len(df))
                df = df.sample(take, random_state=random_state)
            
            else:
                # ensure at least 10 samples per cell type
                grouped = df.groupby("CMT_name")
                sampled_parts = []
                for name, group in grouped:
                    n_total = len(group)
                    if n_total >= 10:
                        sampled = group.copy()
                        print(f"[Info] Kept all {n_total} samples from cell type '{name}'.")
                    else:
                        print(f"[Upsample] cell type '{name}' has only {n_total} samples; upsampling to 10.")
                        sampled = group.sample(n=10, replace=True, random_state=random_state)
                    sampled_parts.append(sampled)
                df = pd.concat(sampled_parts, ignore_index=True)     

            final_df = df.copy()
        else:

            config = self.TASK_CONFIG[downstream_task]
            balance_field = config["balance_field"]
            balance_value = config["balance_value"]
            match_keys = config["match_keys"]

            df = self.last_query_result.copy()
            # ---------- un‑balanced ----------
            if not balanced:
                if downstream_task in ("disease", "gender"):
                    label_col = balance_field
                    df = df[df[label_col].notna() & (df[label_col].astype(str).str.strip() != "")]
                if sample_size:
                    take = min(sample_size, len(df))
                    final_df = df.sample(take, random_state=random_state)
                elif sample_ratio:
                    final_df = df.sample(frac=sample_ratio, random_state=random_state)
                else:
                    final_df = df.copy()
            # ---------- balanced ----------
            else:
                case_df = df.loc[
                    df[balance_field].notna()
                    & (df[balance_field].astype(str).str.strip() != "")
                    & (df[balance_field] != balance_value)
                    & (~df[balance_field].astype(str).str.lower().isin({"unknown", "unannotated", "unannoted", "none", ""}))
                ]
                control_conditions = self.last_query_conditions_resolved.copy()
                control_conditions[self.FIELD_ALIAS.get(balance_field, balance_field)] = balance_value
                control_df = self.df_all.copy()
                for k, v in control_conditions.items():
                    control_df = control_df[control_df[k].isin(v)] if isinstance(v, (list, set, tuple)) else control_df[control_df[k] == v]
                
                control_df = control_df.loc[
                    control_df[balance_field].notna()
                    & (control_df[balance_field].astype(str).str.strip() != "")
                    & (control_df[balance_field] == balance_value)
                    & (~control_df[balance_field].astype(str).str.lower().isin({"unknown", "unannotated", "unannoted", "none", ""}))
                ].copy()

                if len(case_df) == 0 or len(control_df) == 0:
                    raise ValueError(
                        f"No available samples after filtering for task '{downstream_task}': "
                        f"case={len(case_df)}, control={len(control_df)}. "
                        f"Check your query or reduce filters."
                    )

                # Apply sampling before matching
                if sample_size:
                    take = min(sample_size, len(case_df))
                    case_downsample_df = case_df.sample(take, random_state=random_state)
                elif sample_ratio:
                    print(f"[Info] Sampling {sample_ratio * 100:.1f}% of case samples for task '{downstream_task}'.")
                    case_downsample_df = case_df.sample(frac=sample_ratio, random_state=random_state)
                    print(f"[Info] Sampled {len(case_downsample_df)} reference samples for task '{downstream_task}'.")
                else:
                    case_downsample_df = case_df.copy()

                # case samples are less than control samples
                if len(case_downsample_df) <= len(control_df):
                    reference_df = case_downsample_df
                    target_df = control_df
                # case samples are more than control samples
                else:
                    reference_df = control_df
                    target_df = case_downsample_df

                for k in match_keys:
                    reference_df = reference_df[reference_df[k].notna()]
                    target_df    = target_df[target_df[k].notna()]

                matched_target, matched_keys = sample_matched_by_keys(
                    reference_df, target_df, match_keys, random_state=random_state
                )
                # keep only reference rows that have a match
                ref_keep = reference_df[
                    reference_df[match_keys].apply(tuple, axis=1).isin(matched_keys)
                ]

                final_df = pd.concat([ref_keep, matched_target], ignore_index=True)
                print(
                    f"[Info] Matched {len(ref_keep)} reference and {len(matched_target)} target samples for task '{downstream_task}'."
                )

        if shuffle:
            print(f"[Info] Shuffling final DataFrame with {len(final_df)} samples.")
            final_df = final_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        final_df["sample_index"] = np.arange(len(final_df))
        final_df.to_csv(os.path.join(self.root, "last_query_result.csv"), index=False)

        expression_matrix = load_expression_by_metadata(final_df, dataset_dir=self.matrix_root)
        expression_matrix_corrected = dataset_correction(downstream_task, expression_matrix, final_df, output_dir)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            final_df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)
            np.save(os.path.join(output_dir, "expression_matrix.npy"), expression_matrix)
            np.save(os.path.join(output_dir, "expression_matrix_corrected.npy"), expression_matrix_corrected)
            print(f"[Extract] Saved {len(final_df)} samples to '{output_dir}'.")

        return expression_matrix_corrected, final_df
