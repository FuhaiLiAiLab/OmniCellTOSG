import os
import pandas as pd
import numpy as np
from typing import List, Tuple
from .extract import extract_for_inference, extract_for_training

class CellTOSGSubsetBuilder:
    FIELD_ALIAS = {
        "cell_type": "CMT_name",
        "disease": "disease_BMG_name",
        "development_stage": "development_stage_category",
        "sex": "sex_normalized"
    }

    TASK_CONFIG = {
        "pretrain": {},
        "disease": {
            "balance_field": "disease_BMG_name",
            "balance_value": "normal",
            "match_keys": ["CMT_id", "sex_normalized", "development_stage_category"]
        },
        "sex": {
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