import os
import pandas as pd
import numpy as np
from .data_loader import load_expression_by_metadata

def sample_matched_by_keys(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    match_keys,
    stage_col: str = "development_stage_category",
    max_stage_offset: int = 2,
    upsample: bool = True,
    random_state: int = 2025,
):
    """Return two values:

    1. **matched_target_df** – rows from *target_df* picked to mirror *reference_df*.
    2. **matched_keys_set** – the (cell‑type, sex, age‑stage) keys that were *successfully* matched.

    If a key in *reference* has absolutely no corresponding samples in *target* (even
    after stage back‑off) it is considered *unmatched* and is therefore expected to
    be dropped by the caller.
    """

    STAGES = [
        "80 and over", "aged", "middle aged", "adult", "young adult", "adolescent",
        "child", "preschool child", "infant", "newborn", "fetal", "embryonic", "unknown",
    ]
    stage_order = {s: i for i, s in enumerate(STAGES)}

    reference_df = reference_df[reference_df[stage_col].notna()].copy()
    target_df    = target_df[target_df[stage_col].notna()].copy()

    ref_groups = reference_df.groupby(match_keys)
    tgt_groups = target_df.groupby(match_keys)

    matched_parts: list[pd.DataFrame] = []
    matched_keys  = set()

    for key, ref_grp in ref_groups:
        needed = len(ref_grp)
        collected, used_idx = [], set()

        # try same‑stage first
        if key in tgt_groups.groups:
            cand = tgt_groups.get_group(key)
            take = min(len(cand), needed)
            sampled = cand.sample(take, random_state=random_state)
            collected.append(sampled)
            used_idx.update(sampled.index)
            needed -= take

        # back‑off towards younger stages
        stage = key[-1] if isinstance(key, tuple) else key
        if stage not in stage_order:
            continue
        s_idx  = stage_order[stage]
        offset = 1
        while needed > 0 and offset <= max_stage_offset:
            alt_idx = s_idx + offset
            if alt_idx < len(STAGES):
                alt_stage = STAGES[alt_idx]
                alt_key   = (*key[:-1], alt_stage) if isinstance(key, tuple) else alt_stage
                if alt_key in tgt_groups.groups:
                    cand = tgt_groups.get_group(alt_key)
                    cand = cand[~cand.index.isin(used_idx)]
                    take = min(len(cand), needed)
                    if take:
                        sampled = cand.sample(take, random_state=random_state)
                        collected.append(sampled)
                        used_idx.update(sampled.index)
                        needed -= take
            offset += 1

        # up‑sample to fill the gap
        if needed > 0 and upsample and collected:
            pool = pd.concat(collected)
            upsampled = pool.sample(needed, replace=True, random_state=random_state)
            collected.append(upsampled)

        if collected:
            matched_parts.append(pd.concat(collected))
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
            "match_keys": ["CMT_name", "sex_normalized", "development_stage_category"]
        },
        "gender": {
            "balance_field": "sex_normalized",
            "balance_value": "male",
            "match_keys": ["CMT_name", "development_stage_category"]
        }
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

            # Textual fields (case-insensitive)
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
        if self.last_query_result is None:
            raise ValueError("Please call .view() first to select your subset.")

        if downstream_task not in self.TASK_CONFIG:
            raise ValueError(f"Unsupported downstream_task: {downstream_task}")

        config = self.TASK_CONFIG[downstream_task]
        balance_field = config["balance_field"]
        balance_value = config["balance_value"]
        match_keys = config["match_keys"]

        df = self.last_query_result.copy()
         # ---------- un‑balanced ----------
        if not balanced:
            if sample_size:
                final_df = df.sample(sample_size, random_state=random_state)
            elif sample_ratio:
                final_df = df.sample(frac=sample_ratio, random_state=random_state)
            else:
                final_df = df.copy()
        # ---------- balanced ----------
        else:
            case_df = df[df[balance_field] != balance_value]
            control_conditions = self.last_query_conditions_resolved.copy()
            control_conditions[self.FIELD_ALIAS.get(balance_field, balance_field)] = balance_value
            control_df = self.df_all.copy()
            for k, v in control_conditions.items():
                control_df = control_df[control_df[k].isin(v)] if isinstance(v, (list, set, tuple)) else control_df[control_df[k] == v]

            # Apply sampling before matching
            if sample_size:
                case_downsample_df = case_df.sample(sample_size, random_state=random_state)
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
            final_df = final_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        final_df["sample_index"] = np.arange(len(final_df))
        final_df.to_csv(os.path.join(self.root, "last_query_result.csv"), index=False)

        expression_matrix = load_expression_by_metadata(final_df, dataset_dir=self.matrix_root)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            final_df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)
            np.save(os.path.join(output_dir, "expression_matrix.npy"), expression_matrix)
            print(f"[Extract] Saved {len(final_df)} samples to '{output_dir}'.")

        return expression_matrix, final_df
