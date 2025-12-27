# extract.py
import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from .data_loader import load_expression_by_metadata, l1_normalize_log1p, pad_protein_zeros, combat_seq_correction_by_tissue, build_gene_df
from .balancing import balance_for_inference, balance_for_training
from .split import celltype_task_sampling, study_wise_split_pretrain, study_wise_split_without_balancing, study_wise_split_with_balancing

N_PROTEIN = 121_419

def norm_and_export_split(
    self,
    task: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Optional[str],
    shuffle: bool = False,
    random_state: int = 2025,
    target_sum: float = 1e4,
    correction_method: Optional[str] = None,  # None | "combat_seq"
):
    if shuffle:
        train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["sample_index"] = np.arange(len(train_df))
    test_df["sample_index"] = np.arange(len(test_df))

    train_matrix = load_expression_by_metadata(train_df, dataset_dir=self.matrix_root)
    test_matrix = load_expression_by_metadata(test_df, dataset_dir=self.matrix_root)

    # ComBat-Seq correction
    if correction_method is not None:
        method = str(correction_method).strip().lower()
    else:
        method = None

    if method in (None, "", "none"):
        train_corr = train_matrix
        test_corr = test_matrix
    elif method in ("combat", "combat_seq", "pycombat_seq"):
        print("[Info] Applying ComBat-Seq correction.")
        train_corr = combat_seq_correction_by_tissue(
            matrix=train_matrix,
            meta=train_df,
            tissue_col="tissue_general",
            disease_col="disease_BMG_name",
            dataset_id_col="dataset_id",
            fallback_cols=["source", "tissue", "suspension_type", "assay"],
            min_batches_per_group=2,
            min_per_disease=5,
        )

        test_corr = combat_seq_correction_by_tissue(
            matrix=test_matrix,
            meta=test_df,
            tissue_col="tissue_general",
            disease_col="disease_BMG_name",
            dataset_id_col="dataset_id",
            fallback_cols=["source", "tissue", "suspension_type", "assay"],
            min_batches_per_group=2,
            min_per_disease=5,
        )
    else:
        raise ValueError(f"Unsupported correction_method: {correction_method}")

    print(f"shape train_matrix: {train_matrix.shape}, shape test_matrix: {test_matrix.shape}")
    print(f"shape train_corr: {train_corr.shape}, shape test_corr: {test_corr.shape}")

    print(f"first 5 rows of train_matrix:\n{train_matrix[:5, :5]}")
    print(f"first 5 rows of train_corr:\n{train_corr[:5, :5]}")


    train_norm = l1_normalize_log1p(
        matrix=train_corr,
        target_sum=target_sum,
    )

    test_norm = l1_normalize_log1p(
        matrix=test_corr,
        target_sum=target_sum,
    )

    train_norm = pad_protein_zeros(train_norm, n_protein_cols=N_PROTEIN)
    test_norm = pad_protein_zeros(test_norm, n_protein_cols=N_PROTEIN)

    print(f"shape train_norm: {train_norm.shape}, shape test_norm: {test_norm.shape}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        train_dir = os.path.join(output_dir, "train")
        test_dir = os.path.join(output_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        train_df.to_csv(os.path.join(train_dir, "labels.csv"), index=False)
        test_df.to_csv(os.path.join(test_dir, "labels.csv"), index=False)

        # np.save(os.path.join(train_dir, "expression_matrix.npy"), train_matrix)
        # np.save(os.path.join(test_dir, "expression_matrix.npy"), test_matrix)

        # np.save(os.path.join(train_dir, "expression_matrix_norm.npy"), train_norm)
        # np.save(os.path.join(test_dir, "expression_matrix_norm.npy"), test_norm)

        np.save(os.path.join(train_dir, "expression_matrix.npy"), train_norm)
        np.save(os.path.join(test_dir, "expression_matrix.npy"), test_norm)

        print(f"[Export] Saved train ({len(train_df)}) and test ({len(test_df)}) to '{output_dir}'.")

    return (train_norm, train_df), (test_norm, test_df)


def extract_for_inference(
    self,
    task: str = "disease",
    stratified_balancing: bool = False,
    shuffle: bool = False,
    sample_ratio: float | None = None,
    sample_size: int | None = None,
    random_state: int = 2025,
    correction_method: str | None = None,
    output_dir: str | None = None,
):
    """
    Extract expression matrix & labels for modelling (inference mode).

    Parameters
    ----------
    shuffle        Shuffle rows before exporting.
    stratified_balancing       Perform class balancing only for tasks that support it.
    task One of {"disease", "gender", "cell_type"}.
    output_dir     If provided, persist labels + expression to this directory.
    sample_ratio   Fraction (0-1) of rows to keep.
    sample_size    Exact number of rows to keep.
    random_state   Seed for deterministic sampling.
    """
    if self.last_query_result is None:
        return self.available_conditions()

    if task not in self.TASK_CONFIG:
        raise ValueError(f"Unsupported task: {task}")

    if sample_size is not None and sample_ratio is not None:
        print(
            "[Warning] Both sample_size and sample_ratio provided; "
            "sample_ratio will take precedence."
        )
        sample_size = None  # ratio takes precedence

    if task == "cell_type":
        if stratified_balancing:
            print("[Warning] task='cell_type' should not be stratified_balancing, setting stratified_balancing=False.")
            stratified_balancing = False

        df = self.last_query_result.copy()
        # clean CMT_name
        df = df[
            df["CMT_name"].notna()
            & (df["CMT_name"].astype(str).str.strip() != "")
            & (~df["CMT_name"].astype(str).str.lower().isin(
                ["unannoted", "unannotated", "unknown", "miscellaneous", "splatter", "cell"]
            ))
        ].copy()
        if df.empty:
            raise ValueError("No valid rows for cell_type after filtering.")

        # use the unified per-class sampler
        final_df = celltype_task_sampling(
            df=df,
            sample_ratio=sample_ratio,
            sample_size=sample_size,
            random_state=random_state,
            min_per_class=10,
        )
    else:
        # disease / gender
        config = self.TASK_CONFIG[task]
        balance_field = config["balance_field"]
        balance_value = config["balance_value"]
        match_keys = config["match_keys"]

        df = self.last_query_result.copy()

        # Unbalanced path
        if not stratified_balancing:
            if task in ("disease", "gender"):
                label_col = balance_field
                df = df[df[label_col].notna() & (df[label_col].astype(str).str.strip() != "")]
            if sample_size:
                take = min(sample_size, len(df))
                final_df = df.sample(take, random_state=random_state)
            elif sample_ratio:
                final_df = df.sample(frac=sample_ratio, random_state=random_state)
            else:
                final_df = df.copy()

        # Balanced path
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
                    f"No available samples after filtering for task '{task}': "
                    f"case={len(case_df)}, control={len(control_df)}. "
                    f"Check your query or reduce filters."
                )

            # Optional sampling on the case side before matching
            if sample_size:
                take = min(sample_size, len(case_df))
                reference_df = case_df.sample(take, random_state=random_state)
            elif sample_ratio:
                print(f"[Info] Sampling {sample_ratio * 100:.1f}% of case samples for task '{task}'.")
                reference_df = case_df.sample(frac=sample_ratio, random_state=random_state)
                print(f"[Info] Sampled {len(reference_df)} reference samples for task '{task}'.")
            else:
                reference_df = case_df.copy()

            # Choose the smaller split as reference
            if len(reference_df) <= len(control_df):
                target_df = control_df
            else:
                target_df = reference_df
                reference_df = control_df

            # Drop rows missing any match key
            for k in match_keys:
                reference_df = reference_df[reference_df[k].notna()]
                target_df = target_df[target_df[k].notna()]

            # Use inference balancing
            matched_target, matched_keys = balance_for_inference(
                reference_df=reference_df,
                target_df=target_df,
                match_keys=match_keys,
                random_state=random_state,
            )

            # Keep only reference rows that actually matched
            ref_keep = reference_df[
                reference_df[match_keys].apply(tuple, axis=1).isin(matched_keys)
            ]

            final_df = pd.concat([ref_keep, matched_target], ignore_index=True)
            print(
                f"[Info] Matched {len(ref_keep)} reference and {len(matched_target)} target samples "
                f"for task '{task}'."
            )

    if shuffle:
        print(f"[Info] Shuffling final DataFrame with {len(final_df)} samples.")
        final_df = final_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    final_df["sample_index"] = np.arange(len(final_df))
    final_df.to_csv(os.path.join(self.root, "last_query_result.csv"), index=False)

    expression_matrix = load_expression_by_metadata(final_df, dataset_dir=self.matrix_root)
    
    gene_df_raw, gene_df_corr, final_df, choice_df = build_gene_df(
        final_df=final_df,
        expression_matrix=expression_matrix,
        bmg_gene_index_csv=self.bmg_gene_index_csv,
        n_col_expected=412039,
        correction_method=correction_method,
    )

    print(f"gene_df_raw shape: {gene_df_raw.shape}")
    print(f"gene_df_corr shape: {gene_df_corr.shape}")
    print(f"first 5x5 gene_df_raw:\n{gene_df_raw.iloc[:5, :5]}")
    print(f"first 5x5 gene_df_corr:\n{gene_df_corr.iloc[:5, :5]}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        final_df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)
        choice_df.to_csv(os.path.join(output_dir, "bmg_to_gene_choice.csv"), index=False)

        gene_df_raw.to_csv(os.path.join(output_dir, "expression_gene.csv"))

        if correction_method is not None:
            gene_df_corr.to_csv(os.path.join(output_dir, "expression_gene_corrected.csv"))

        print(f"[Extract] Saved {len(final_df)} samples to '{output_dir}'.")

    gene_df_out = gene_df_corr if gene_df_corr is not None else gene_df_raw

    return gene_df_out, final_df


def extract_for_training(
    self,
    task: str = "disease",
    stratified_balancing: bool = False,
    shuffle: bool = False,
    sample_ratio: float | None = None,
    sample_size: int | None = None,
    random_state: int = 2025,
    study_test_ratio: float = 0.2,
    correction_method: str | None = None,
    output_dir: str | None = None,
):
    if self.last_query_result is None:
        return self.available_conditions()

    # pretrain task
    if task == "pretrain":
        if stratified_balancing:
            print("[Warning] task='pretrain' does not accept stratified balancing, setting stratified_balancing=False.")
            stratified_balancing = False

        train_df, test_df, test_studies = study_wise_split_pretrain(
            self,
            sample_ratio=sample_ratio,
            random_state=random_state,
            test_fraction_by_samples=0.20,
            hard_cap_fraction_by_samples=0.40,
            ensure_min_per_group=10,
        )

        # Export with correction
        return norm_and_export_split(
            self=self,
            task=task,
            train_df=train_df,
            test_df=test_df,
            output_dir=output_dir,
            shuffle=shuffle,
            random_state=random_state,
            correction_method=correction_method,
        )
    # end pretrain

    if "dataset_id" not in self.last_query_result.columns:
        raise KeyError("Column 'dataset_id' is required for study-wise split in training mode.")
    if study_test_ratio <= 0 or study_test_ratio >= 1:
        raise ValueError("study_test_ratio must be in (0, 1).")

    # Prepare reference_df per task
    ref_df = self.last_query_result.copy()
    if task == "cell_type":
        if stratified_balancing:
            print("[Warning] task='cell_type' should not use stratified balancing, setting stratified_balancing=False.")
            stratified_balancing = False
        category_col = "CMT_name"
        ref_df = ref_df[
            ref_df["CMT_name"].notna() # remove NAs
            & (ref_df["CMT_name"].astype(str).str.strip() != "") # remove empty strings
            & (~ref_df["CMT_name"].astype(str).str.lower().isin(
                ["unannoted", "unannotated", "unknown", "miscellaneous", "splatter", "cell"]
            )) # remove low-quality cell types
        ].copy()
        if ref_df.empty:
            raise ValueError("No valid rows for cell_type after filtering.")
    else:
        category_col = self.TASK_CONFIG[task]["balance_field"]
        ref_df = ref_df[ref_df[category_col].notna() & (ref_df[category_col].astype(str).str.strip() != "")]
        if ref_df.empty:
            raise ValueError(f"No valid rows for task '{task}' after filtering '{category_col}'.")

    # Split reference into train/test by studies
    if not stratified_balancing:
        if task == "cell_type":
            train, test, test_studies, forced_train = study_wise_split_without_balancing(
                reference_df=ref_df,
                category_col=category_col,
                study_col="dataset_id",
                test_fraction_by_samples=0.20,
                hard_cap_fraction_by_samples=0.40,
                random_state=random_state,
            )

            # Downsample after split (per split)
            train = celltype_task_sampling(
                df=train,
                sample_ratio=sample_ratio,
                sample_size=sample_size,
                random_state=random_state,
                min_per_class=10,
            )
            test = celltype_task_sampling(
                df=test,
                sample_ratio=sample_ratio,
                sample_size=sample_size,
                random_state=random_state,
                min_per_class=10,
            )

            # Export with correction
            return norm_and_export_split(
                self=self,
                task=task,
                train_df=train,
                test_df=test,
                output_dir=output_dir,
                shuffle=shuffle,
                random_state=random_state,
                correction_method=correction_method,
            )
        else:
            raise ValueError("Other train tasks require stratified_balancing=True.")
        
    elif stratified_balancing:

        # build reference_df / target_df
        config = self.TASK_CONFIG[task]
        balance_field = config["balance_field"]
        balance_value = config["balance_value"]
        match_keys = config["match_keys"]

        # case: label != balance_value & valid
        case_df = ref_df.loc[
            ref_df[balance_field].notna()
            & (ref_df[balance_field].astype(str).str.strip() != "")
            & (ref_df[balance_field] != balance_value)
            & (~ref_df[balance_field].astype(str).str.lower().isin({"unknown", "unannotated", "unannoted", "none", ""}))
        ].copy()

        # control: use view() conditions + set balance_field = balance_value, then filter valid & == balance_value
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
                f"No available samples after filtering for task '{task}': "
                f"case={len(case_df)}, control={len(control_df)}. "
                f"Check your query or reduce filters."
            )

        # Choose smaller as reference, larger as target
        if len(case_df) <= len(control_df):
            reference_df = case_df
            target_df = control_df
        else:
            reference_df = control_df
            target_df = case_df

        # study-wise split + joint balancing
        train_balanced, test_balanced, tgt_train, tgt_test, test_studies, forced_train = study_wise_split_with_balancing(
            reference_df=reference_df,
            target_df=target_df,
            match_keys=match_keys,
            study_col="dataset_id",
            test_fraction_by_samples=0.20,
            hard_cap_fraction_by_samples=0.40,
            random_state=random_state,
        )
        # Downsampling AFTER balancing (per split)
        if sample_ratio is not None:
            train_balanced = train_balanced.sample(frac=sample_ratio, random_state=random_state)
            test_balanced  = test_balanced.sample(frac=sample_ratio,  random_state=random_state)
        elif sample_size is not None:
            take_tr = min(sample_size, len(train_balanced))
            take_te = min(sample_size, len(test_balanced))
            train_balanced = train_balanced.sample(n=take_tr, random_state=random_state)
            test_balanced  = test_balanced.sample(n=take_te, random_state=random_state)

        # Export with correction
        return norm_and_export_split(
            self=self,
            task=task,
            train_df=train_balanced,
            test_df=test_balanced,
            output_dir=output_dir,
            shuffle=shuffle,
            random_state=random_state,
            correction_method=correction_method,
        )
    else:
        raise ValueError("Invalid stratified_balancing value.")