# split_helpers.py
import math
import numpy as np
import pandas as pd
from typing import Tuple, List
from .balancing import balance_for_training

def celltype_task_sampling(
    df: pd.DataFrame,
    sample_ratio: float | None,
    sample_size: int | None,
    random_state: int,
    min_per_class: int = 10,
) -> pd.DataFrame:
    """
    Per-class sampling for cell_type task.
    Ensures each CMT_name has at least `min_per_class` rows (upsample with replace if needed).

    Rules
    -----
    - If sample_ratio: for each class take max(floor(n*ratio), min_per_class).
    - If sample_size: enforce >= min_per_class per class; distribute remaining by proportion.
      If sample_size < min_per_class * #classes -> error.
    - Else: keep all; upsample classes with < min_per_class to min_per_class.
    """
    if "CMT_name" not in df.columns:
        raise KeyError("Column 'CMT_name' is required for cell_type sampling.")

    grouped = list(df.groupby("CMT_name"))
    classes = [name for name, _ in grouped]
    counts  = {name: len(g) for name, g in grouped}

    if sample_ratio is not None:
        parts = []
        for name, g in grouped:
            n_total = len(g)
            n_take  = max(int(n_total * sample_ratio), min_per_class)
            sampled = (
                g.sample(n=n_take, replace=False, random_state=random_state)
                if n_take <= n_total else
                g.sample(n=n_take, replace=True,  random_state=random_state)
            )
            parts.append(sampled)
        return pd.concat(parts, ignore_index=True)

    if sample_size is not None:
        C = len(classes)
        min_needed = min_per_class * C
        if sample_size < min_needed:
            raise ValueError(
                f"sample_size={sample_size} is too small to ensure at least "
                f"{min_per_class} per class across {C} classes (>= {min_needed} required)."
            )

        N = sum(counts.values()) or 1  # avoid div-by-zero
        # initial quotas (lower bound = min_per_class)
        initial = {}
        rema = {}
        for name in classes:
            target_prop = sample_size * (counts[name] / N)
            q_floor = int(np.floor(target_prop))
            initial[name] = max(min_per_class, q_floor)
            rema[name] = target_prop - q_floor

        total_q = sum(initial.values())

        # adjust down if exceeding
        if total_q > sample_size:
            over = total_q - sample_size
            adjustable = sorted(
                [n for n in classes if initial[n] > min_per_class],
                key=lambda n: initial[n] - min_per_class,
                reverse=True,
            )
            idx = 0
            while over > 0 and adjustable:
                name = adjustable[idx % len(adjustable)]
                if initial[name] > min_per_class:
                    initial[name] -= 1
                    over -= 1
                else:
                    adjustable.pop(idx % len(adjustable))
                idx += 1

        # adjust up if under
        elif total_q < sample_size:
            under = sample_size - total_q
            by_rema = sorted(classes, key=lambda n: rema[n], reverse=True)
            idx = 0
            while under > 0 and by_rema:
                name = by_rema[idx % len(by_rema)]
                initial[name] += 1
                under -= 1
                idx += 1

        # final sampling
        parts = []
        for name, g in grouped:
            n_total = len(g)
            n_take  = initial[name]
            sampled = (
                g.sample(n=n_take, replace=False, random_state=random_state)
                if n_take <= n_total else
                g.sample(n=n_take, replace=True,  random_state=random_state)
            )
            parts.append(sampled)
        return pd.concat(parts, ignore_index=True)

    # ensure >= min_per_class per class
    parts = []
    for name, g in grouped:
        n_total = len(g)
        sampled = g.copy() if n_total >= min_per_class else g.sample(n=min_per_class, replace=True, random_state=random_state)
        parts.append(sampled)
    return pd.concat(parts, ignore_index=True)


def study_wise_split_pretrain(
    self,
    sample_ratio: float | None = None,
    random_state: int = 2025,
    test_fraction_by_samples: float = 0.20,   # target ~20% of samples
    hard_cap_fraction_by_samples: float = 0.40,  # no exceed 40%
    ensure_min_per_group: int = 10,           # min per (dataset_id, disease_BMG_name) after sampling
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """

    1) Filter is_pretrain_data == True from self.last_query_result.
    2) Per-group sampling by (dataset_id, disease_BMG_name) with a minimum of ensure_min_per_group.
       - If sample_ratio is None: take all.
       - Else: take floor(n * sample_ratio), but at least ensure_min_per_group if possible; if group size < ensure_min_per_group, take all.
    3) Choose test studies by sample counts:
       - Sort studies by count desc; pick from the tail (smallest studies first) until reaching ~test_fraction_by_samples of total samples;
       - If still under target, try adding one more smallest study without exceeding the hard cap;
       - Ensure test contains at least 1 study (if unique studies < 2, raise).
    4) Return train_df, test_df, and the list of test_studies.

    """
    df = self.last_query_result.copy()

    if "is_pretrain_data" not in df.columns:
        raise KeyError("Column 'is_pretrain_data' not found in metadata for pretrain task.")
    df = df[df["is_pretrain_data"].fillna(False)]
    if df.empty:
        raise ValueError("No samples with is_pretrain_data=True under current filters.")

    for col in ("dataset_id", "disease_BMG_name"):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in metadata for pretrain grouping.")

    grouped = df.groupby(["dataset_id", "disease_BMG_name"], dropna=False)

    ratio = 1.0 if sample_ratio is None else float(sample_ratio)
    if ratio <= 0:
        raise ValueError("sample_ratio must be > 0 for pretrain task.")

    sampled_parts = []
    for (ds, dis), group in grouped:
        n = len(group)
        n_take = n if ratio >= 1.0 else int(np.floor(n * ratio))
        if n_take < ensure_min_per_group:
            if n <= ensure_min_per_group:
                sampled = group.copy()  # take all
            else:
                sampled = group.sample(n=ensure_min_per_group, replace=False, random_state=random_state)
        else:
            n_take = min(n_take, n)
            sampled = group.sample(n=n_take, replace=False, random_state=random_state)
        sampled_parts.append(sampled)

    if not sampled_parts:
        raise ValueError("No groups for pretrain task after sampling.")

    final_df = pd.concat(sampled_parts, ignore_index=True)

    if "dataset_id" not in final_df.columns:
        raise KeyError("Column 'dataset_id' is required for study-wise split in pretrain task.")

    # Study-wise selection by sample counts
    study_counts = final_df["dataset_id"].astype(str).value_counts().sort_values(ascending=False)
    unique_studies = study_counts.index.tolist()
    if len(unique_studies) < 2:
        raise ValueError("Study-wise split requires at least 2 distinct dataset_id.")

    total_n = int(study_counts.sum())
    target_min = int(np.floor(total_n * test_fraction_by_samples))
    hard_max  = int(np.floor(total_n * hard_cap_fraction_by_samples))

    test_studies, test_n = [], 0
    for ds in reversed(unique_studies):  # from smallest upwards
        c = int(study_counts.loc[ds])
        if test_n + c <= target_min:
            test_studies.append(ds)
            test_n += c
        else:
            break

    # If still < target, try to add one more smallest study without exceeding hard cap
    remaining = [ds for ds in reversed(unique_studies) if ds not in test_studies]
    if test_n < target_min and remaining:
        ds_extra = remaining[0]
        c_extra = int(study_counts.loc[ds_extra])
        if test_n + c_extra <= hard_max:
            test_studies.append(ds_extra)
            test_n += c_extra
        # else keep as-is

    if not test_studies:
        ds_smallest = unique_studies[-1]
        c_smallest = int(study_counts.loc[ds_smallest])
        if c_smallest == 0:
            raise ValueError("Cannot form a non-empty test split.")
        test_studies = [ds_smallest]

    # Final split (no shuffle)
    test_mask = final_df["dataset_id"].astype(str).isin(set(test_studies))
    test_df = final_df[test_mask].copy()
    train_df = final_df[~test_mask].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(f"Empty split after study-wise partition: train={len(train_df)}, test={len(test_df)}.")

    return train_df, test_df, test_studies


def study_wise_split_without_balancing(
    reference_df: pd.DataFrame,
    category_col: str,                 # e.g., "CMT_name" for cell_type, or balance_field for disease/gender
    study_col: str = "dataset_id",
    test_fraction_studies: float = 0.20,
    random_state: int = 2025,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Split reference_df into train, test sets at study level. If a category (e.g., a cell type) exists ONLY in a single study, that study goes to train.

    --------
    - Sort studies by size (desc).
    - Compute 'must-train' studies: studies that contain any category exclusive to that study.
    - From the remaining studies (desc order), pick ceil(20% of total studies) as test.
      If that would result in zero test studies, pick the first remaining one.
    - Return train_df and test_df, along with chosen test_studies and forced_train_studies.

    """
    if study_col not in reference_df.columns:
        raise KeyError(f"Column '{study_col}' not found in reference_df.")
    if category_col not in reference_df.columns:
        raise KeyError(f"Column '{category_col}' not found in reference_df.")

    # Count samples per study and sort desc
    study_sizes = reference_df[study_col].astype(str).value_counts().sort_values(ascending=False)
    studies_desc = study_sizes.index.tolist()
    if len(studies_desc) < 2:
        raise ValueError("At least 2 distinct studies are required for a train/test split.")

    # Map category -> set of studies that contain it
    cat_study_map = (
        reference_df[[study_col, category_col]]
        .dropna(subset=[study_col, category_col])
        .astype({study_col: str, category_col: str})
        .drop_duplicates()
        .groupby(category_col)[study_col]
        .apply(set)
    )

    # Studies that contain any category that appears in exactly one study
    exclusive_cats = [cat for cat, s in cat_study_map.items() if len(s) == 1]
    forced_train_studies: set = set()
    for cat in exclusive_cats:
        forced_train_studies |= cat_study_map[cat]

    # Desired number of test studies (by count, not by samples)
    desired_test_count = max(1, math.ceil(len(studies_desc) * float(test_fraction_studies)))

    # Candidate studies that are NOT forced to train
    remaining = [s for s in studies_desc if s not in forced_train_studies]

    if not remaining:
        # All studies are forced into train by exclusivity; cannot form a leakage-free test set
        raise ValueError("All studies contain exclusive categories; cannot form a test split without leakage.")

    # Pick top 'desired_test_count' studies from the remaining (desc order)
    test_studies = remaining[:desired_test_count]
    train_studies = [s for s in studies_desc if s not in test_studies]

    # Make DataFrames
    test_mask = reference_df[study_col].astype(str).isin(set(test_studies))
    test_df = reference_df[test_mask].copy()
    train_df = reference_df[~test_mask].copy()

    if test_df.empty or train_df.empty:
        raise ValueError(
            f"Empty split after study-wise partition: train={len(train_df)}, test={len(test_df)}. "
            "Adjust data or the splitting rule."
        )

    return train_df, test_df, test_studies, sorted(forced_train_studies)


def study_wise_split_with_balancing(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    category_col: str,
    study_col: str = "dataset_id",
    test_fraction_studies: float = 0.20,
    random_state: int = 2025,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:

    # validations
    if study_col not in reference_df.columns:
        raise KeyError(f"Column '{study_col}' not found in reference_df.")
    if study_col not in target_df.columns:
        raise KeyError(f"Column '{study_col}' not found in target_df.")
    if category_col not in reference_df.columns:
        raise KeyError(f"Column '{category_col}' not found in reference_df.")

    # study sizes (desc)
    study_sizes = reference_df[study_col].astype(str).value_counts().sort_values(ascending=False)
    studies_desc = study_sizes.index.tolist()
    if len(studies_desc) < 2:
        raise ValueError("At least 2 distinct studies are required for a train/test split.")

    # compute forced_train by exclusivity on stage/sex if available
    forced_train_studies: set = set()
    for guard_col in ["development_stage_category", "sex_normalized"]:
        if guard_col in reference_df.columns:
            # value -> set of studies
            val_study_map = (
                reference_df[[study_col, guard_col]]
                .dropna(subset=[study_col, guard_col])
                .astype({study_col: str, guard_col: str})
                .drop_duplicates()
                .groupby(guard_col)[study_col]
                .apply(set)
            )
            # any value that appears in exactly one study forces that study into train
            for _, studies in val_study_map.items():
                if len(studies) == 1:
                    forced_train_studies |= studies

    # pick ~20%  remaining smallest studies for test
    desired_test_count = max(1, int(np.ceil(len(studies_desc) * float(test_fraction_studies))))
    remaining = [s for s in studies_desc if s not in forced_train_studies]
    if not remaining:
        raise ValueError("All studies are forced into train by exclusivity; cannot form a leakage-free test set.")

    # from the tail (smallest) pick desired_test_count
    remaining_tail_smallest = list(reversed(remaining))
    test_studies = remaining_tail_smallest[:desired_test_count]

    # ensure at least one test study
    if not test_studies:
        # take the smallest non-forced study
        test_studies = [remaining_tail_smallest[0]]

    # split reference
    test_mask = reference_df[study_col].astype(str).isin(set(test_studies))
    ref_test = reference_df[test_mask].copy()
    ref_train = reference_df[~test_mask].copy()
    if ref_train.empty or ref_test.empty:
        raise ValueError(
            f"Empty split after study-wise partition: train={len(ref_train)}, test={len(ref_test)}. "
            "Adjust data or the splitting rule."
        )

    # split target for inspection/debug (not strictly required)
    tgt_test_mask = target_df[study_col].astype(str).isin(set(test_studies))
    tgt_test = target_df[tgt_test_mask].copy()
    tgt_train = target_df[~tgt_test_mask].copy()

    # balancing
    train_balanced, test_balanced = balance_for_training(
        ref_train=ref_train,
        ref_test=ref_test,
        target_df=target_df,
        category_col=category_col,
        study_col=study_col,
        random_state=random_state,
    )

    # Return balanced splits plus raw target splits and bookkeeping lists
    return train_balanced, test_balanced, tgt_train, tgt_test, test_studies, sorted(forced_train_studies)
