# extract.py
import os
import numpy as np
import pandas as pd
from .data_loader import load_expression_by_metadata, dataset_correction
from .balancing import balance_for_inference, balance_for_training
from .split import celltype_task_sampling, study_wise_split_pretrain, study_wise_split_without_balancing, study_wise_split_with_balancing

def correct_and_export_split(
    self,
    task: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | None,
    shuffle: bool = False,
    random_state: int = 2025,
):

    if shuffle:
        train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        test_df  = test_df.sample(frac=1,  random_state=random_state).reset_index(drop=True)

    # Sample indices
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["sample_index"] = np.arange(len(train_df))
    test_df["sample_index"]  = np.arange(len(test_df))

    # Raw matrices (row order must match DataFrame)
    train_matrix = load_expression_by_metadata(train_df, dataset_dir=self.matrix_root)
    test_matrix  = load_expression_by_metadata(test_df,  dataset_dir=self.matrix_root)

    try:
        train_corr, test_corr = dataset_correction(
            task,
            train_matrix, train_df,
            output_dir,
            split=True,
            test_matrix=test_matrix,
            test_meta=test_df,
        )
    except TypeError:
        print("[Warning] dataset_correction(split=True, ...) not available; falling back to single-matrix correction (may cause leakage).")
        all_df = pd.concat([train_df, test_df], ignore_index=True)
        all_matrix = load_expression_by_metadata(all_df, dataset_dir=self.matrix_root)
        all_corr = dataset_correction(task, all_matrix, all_df, output_dir)
        n_tr = len(train_df)
        train_corr, test_corr = all_corr[:n_tr], all_corr[n_tr:]

    # Save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        train_dir = os.path.join(output_dir, "train")
        test_dir  = os.path.join(output_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir,  exist_ok=True)

        # labels
        train_df.to_csv(os.path.join(train_dir, "labels.csv"), index=False)
        test_df.to_csv(os.path.join(test_dir, "labels.csv"), index=False)

        # raw matrices
        np.save(os.path.join(train_dir, "expression_matrix.npy"), train_matrix)
        np.save(os.path.join(test_dir,  "expression_matrix.npy"), test_matrix)

        # corrected matrices
        np.save(os.path.join(train_dir, "expression_matrix_corrected.npy"), train_corr)
        np.save(os.path.join(test_dir,  "expression_matrix_corrected.npy"), test_corr)

        print(f"[Export] Saved train ({len(train_df)}) and test ({len(test_df)}) to '{output_dir}'.")

    return (train_corr, train_df), (test_corr, test_df)


def extract_for_inference(
    self,
    task: str = "disease",
    stratified_balancing: bool = False,
    shuffle: bool = False,
    sample_ratio: float | None = None,
    sample_size: int | None = None,
    random_state: int = 2025,
    dataset_correction: str | None = None,
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
        raise ValueError("Please call .view() first to select your subset.")

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
    
    if dataset_correction is not None:
        expression_matrix_corrected = dataset_correction(task, expression_matrix, final_df, dataset_correction, output_dir)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        final_df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)
        np.save(os.path.join(output_dir, "expression_matrix.npy"), expression_matrix)
        if dataset_correction is not None:
            np.save(os.path.join(output_dir, "expression_matrix_corrected.npy"), expression_matrix_corrected)
        print(f"[Extract] Saved {len(final_df)} samples to '{output_dir}'.")

    if dataset_correction is not None:
        return expression_matrix_corrected, final_df
    else:
        return expression_matrix, final_df


def extract_for_training(
    self,
    task: str = "disease",
    stratified_balancing: bool = False,
    shuffle: bool = False,
    sample_ratio: float | None = None,
    sample_size: int | None = None,
    random_state: int = 2025,
    study_test_ratio: float = 0.2,
    output_dir: str | None = None,
):
    if self.last_query_result is None:
        raise ValueError("Please call .view() first to select your subset.")

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
        return correct_and_export_split(
            self=self,
            task=task,
            train_df=train_df,
            test_df=test_df,
            output_dir=output_dir,
            shuffle=shuffle,
            random_state=random_state,
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
                test_fraction_studies=study_test_ratio,
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
            return correct_and_export_split(
                self=self,
                task=task,
                train_df=train,
                test_df=test,
                output_dir=output_dir,
                shuffle=shuffle,
                random_state=random_state,
            )
        else:
            raise ValueError("Other train tasks require stratified_balancing=True.")
        
    elif stratified_balancing:

        # build reference_df / target_df
        config = self.TASK_CONFIG[task]
        balance_field = config["balance_field"]
        balance_value = config["balance_value"]

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
        train_balanced, test_balanced, test_studies, forced_train = study_wise_split_with_balancing(
            reference_df=reference_df,
            target_df=target_df,
            category_col=category_col,
            study_col="dataset_id",
            test_fraction_studies=study_test_ratio,
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
        return correct_and_export_split(
            self=self,
            task=task,
            train_df=train_balanced,
            test_df=test_balanced,
            output_dir=output_dir,
            shuffle=shuffle,
            random_state=random_state,
        )
    else:
        raise ValueError("Invalid stratified_balancing value.")