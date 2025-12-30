# split_helpers.py
import math
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Set
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


def donor_wise_split_pretrain(
    self,
    sample_ratio: float | None = None,
    random_state: int = 2025,
    test_fraction_by_samples: float = 0.20,
    hard_cap_fraction_by_samples: float = 0.40,
    ensure_min_per_group: int = 10,  # min per (dataset_id, disease_BMG_name) after sampling
    study_col: str = "dataset_id",
    donor_col: str = "donor_id",
    min_donor_keys_required_for_split: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:

    df = self.last_query_result.copy()

    if "is_pretrain_data" not in df.columns:
        raise KeyError("Column 'is_pretrain_data' not found in metadata for pretrain task.")
    df = df[df["is_pretrain_data"].fillna(False)]
    if df.empty:
        raise ValueError("No samples with is_pretrain_data=True under current filters.")

    required_cols = [study_col, donor_col, "disease_BMG_name"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in metadata for pretrain task.")

    # Clean ids
    df[study_col] = df[study_col].astype(str).str.strip()
    df[donor_col] = df[donor_col].astype(str).str.strip()
    df = df[df[donor_col].notna() & (df[donor_col].astype(str).str.strip() != "")]
    if df.empty:
        raise ValueError("No samples left after filtering invalid donor_id in pretrain task.")

    # Step 2: per-group sampling
    grouped = df.groupby([study_col, "disease_BMG_name"], dropna=False)

    ratio = 1.0 if sample_ratio is None else float(sample_ratio)
    if ratio <= 0:
        raise ValueError("sample_ratio must be > 0 for pretrain task.")

    sampled_parts: list[pd.DataFrame] = []
    for _, group in grouped:
        n = len(group)
        n_take = n if ratio >= 1.0 else int(np.floor(n * ratio))

        if n_take < ensure_min_per_group:
            if n <= ensure_min_per_group:
                sampled = group.copy()
            else:
                sampled = group.sample(n=ensure_min_per_group, replace=False, random_state=random_state)
        else:
            n_take = min(n_take, n)
            sampled = group.sample(n=n_take, replace=False, random_state=random_state)

        sampled_parts.append(sampled)

    if not sampled_parts:
        raise ValueError("No groups for pretrain task after sampling.")

    final_df = pd.concat(sampled_parts, ignore_index=True)

    # donor_key
    final_df["donor_key"] = final_df[study_col] + "||" + final_df[donor_col]

    donor_counts = final_df["donor_key"].value_counts().sort_values(ascending=False)
    donor_keys_desc = donor_counts.index.tolist()

    if len(donor_keys_desc) < min_donor_keys_required_for_split:
        raise ValueError(
            f"Donor-wise split requires at least {min_donor_keys_required_for_split} distinct donor_key; "
            f"got {len(donor_keys_desc)}."
        )

    total_n = int(donor_counts.sum())
    target_min = int(math.floor(total_n * float(test_fraction_by_samples)))
    hard_max = int(math.floor(total_n * float(hard_cap_fraction_by_samples)))

    if target_min <= 0:
        raise ValueError("test_fraction_by_samples too small; target test sample count is 0.")
    if hard_max <= 0:
        raise ValueError("hard_cap_fraction_by_samples too small; hard cap test sample count is 0.")
    if hard_max >= total_n:
        raise ValueError("hard_cap_fraction_by_samples must keep hard cap < total samples.")

    # Pick from smallest donors first (tie-broken by shuffle for repeatability)
    rng = np.random.RandomState(random_state)
    donor_keys = donor_keys_desc.copy()
    rng.shuffle(donor_keys)
    donor_keys_small_to_large = sorted(donor_keys, key=lambda k: int(donor_counts.loc[k]))

    test_donor_keys: List[str] = []
    test_n = 0

    for dk in donor_keys_small_to_large:
        c = int(donor_counts.loc[dk])
        if test_n + c <= target_min:
            test_donor_keys.append(dk)
            test_n += c
        else:
            break

    remaining_after = [dk for dk in donor_keys_small_to_large if dk not in test_donor_keys]
    if test_n < target_min and remaining_after:
        dk_extra = remaining_after[0]
        c_extra = int(donor_counts.loc[dk_extra])
        if test_n + c_extra <= hard_max:
            test_donor_keys.append(dk_extra)
            test_n += c_extra

    if not test_donor_keys:
        test_donor_keys = [donor_keys_small_to_large[0]]

    # Final split
    test_mask = final_df["donor_key"].isin(set(test_donor_keys))
    test_df = final_df[test_mask].copy()
    train_df = final_df[~test_mask].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Empty split after donor-wise partition: train={len(train_df)}, test={len(test_df)}."
        )

    # Cleanup helper
    train_df = train_df.drop(columns=["donor_key"])
    test_df = test_df.drop(columns=["donor_key"])

    return train_df, test_df, test_donor_keys


def donor_wise_split_without_balancing(
    reference_df: pd.DataFrame,
    category_col: str,                 # e.g., "CMT_name" for cell_type
    study_col: str = "dataset_id",
    donor_col: str = "donor_id",
    test_fraction_by_samples: float = 0.20,
    hard_cap_fraction_by_samples: float = 0.40,
    min_samples_per_category_per_split: int = 1,
    min_donors_per_category_per_split: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    if study_col not in reference_df.columns:
        raise KeyError(f"Column '{study_col}' not found in reference_df.")
    if donor_col not in reference_df.columns:
        raise KeyError(f"Column '{donor_col}' not found in reference_df.")
    if category_col not in reference_df.columns:
        raise KeyError(f"Column '{category_col}' not found in reference_df.")

    if test_fraction_by_samples <= 0 or test_fraction_by_samples >= 1:
        raise ValueError("test_fraction_by_samples must be in (0, 1).")
    if hard_cap_fraction_by_samples <= 0 or hard_cap_fraction_by_samples >= 1:
        raise ValueError("hard_cap_fraction_by_samples must be in (0, 1).")
    if hard_cap_fraction_by_samples < test_fraction_by_samples:
        raise ValueError("hard_cap_fraction_by_samples must be >= test_fraction_by_samples.")

    df = reference_df.copy()
    df[study_col] = df[study_col].astype(str).str.strip()
    df[donor_col] = df[donor_col].astype(str).str.strip()
    df[category_col] = df[category_col].astype(str)

    # drop rows without donor info (recommended for leakage safety)
    df = df[df[donor_col].notna() & (df[donor_col].astype(str).str.strip() != "")]
    if df.empty:
        raise ValueError("No rows left after filtering invalid donor_id.")

    df["donor_key"] = df[study_col] + "||" + df[donor_col]

    donor_sizes = df["donor_key"].value_counts()
    donor_keys_all = donor_sizes.index.tolist()
    if len(donor_keys_all) < 2:
        raise ValueError("At least 2 distinct donor_key are required for a train/test split.")

    total_samples = int(len(df))
    target_min = int(math.floor(total_samples * float(test_fraction_by_samples)))
    hard_max = int(math.floor(total_samples * float(hard_cap_fraction_by_samples)))
    if target_min <= 0:
        raise ValueError("test_fraction_by_samples is too small; target test sample count is 0.")
    if hard_max <= 0:
        raise ValueError("hard_cap_fraction_by_samples is too small; hard cap test sample count is 0.")
    if hard_max >= total_samples:
        raise ValueError("hard_cap_fraction_by_samples must keep hard cap < total samples.")

    # pick test donors by accumulating smallest donors first
    donor_keys_small_to_large = sorted(donor_keys_all, key=lambda k: int(donor_sizes.loc[k]))

    test_donor_keys: List[str] = []
    test_n = 0
    for k in donor_keys_small_to_large:
        c = int(donor_sizes.loc[k])
        if test_n + c <= target_min:
            test_donor_keys.append(k)
            test_n += c
        else:
            break

    remaining_after = [k for k in donor_keys_small_to_large if k not in test_donor_keys]
    if test_n < target_min and remaining_after:
        k_extra = remaining_after[0]
        c_extra = int(donor_sizes.loc[k_extra])
        if test_n + c_extra <= hard_max:
            test_donor_keys.append(k_extra)
            test_n += c_extra

    if not test_donor_keys:
        test_donor_keys = [donor_keys_small_to_large[0]]

    test_mask = df["donor_key"].isin(set(test_donor_keys))
    test_df = df[test_mask].copy()
    train_df = df[~test_mask].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(f"Empty split: train={len(train_df)}, test={len(test_df)}.")

    # Drop labels not covered on both sides (by donor + sample thresholds)
    def label_stats(d: pd.DataFrame) -> pd.DataFrame:
        return (
            d.groupby(category_col)
            .agg(
                n_samples=(category_col, "size"),
                n_donors=("donor_key", "nunique"),
            )
            .reset_index()
        )

    tr_stats = label_stats(train_df).set_index(category_col)
    te_stats = label_stats(test_df).set_index(category_col)

    labels_all = sorted(set(tr_stats.index) | set(te_stats.index))
    keep_labels: List[str] = []

    for lab in labels_all:
        tr_n = int(tr_stats.loc[lab, "n_samples"]) if lab in tr_stats.index else 0
        te_n = int(te_stats.loc[lab, "n_samples"]) if lab in te_stats.index else 0
        tr_d = int(tr_stats.loc[lab, "n_donors"]) if lab in tr_stats.index else 0
        te_d = int(te_stats.loc[lab, "n_donors"]) if lab in te_stats.index else 0

        ok = (
            tr_n >= min_samples_per_category_per_split
            and te_n >= min_samples_per_category_per_split
            and tr_d >= min_donors_per_category_per_split
            and te_d >= min_donors_per_category_per_split
        )
        if ok:
            keep_labels.append(lab)

    if not keep_labels:
        raise ValueError(
            "After donor-wise split, no labels satisfy coverage constraints on both train and test. "
            "Try lowering thresholds or adding more data."
        )

    train_df = train_df[train_df[category_col].isin(set(keep_labels))].copy()
    test_df = test_df[test_df[category_col].isin(set(keep_labels))].copy()

    # Final safety: test labels must be subset of train labels
    train_labels = set(train_df[category_col].astype(str))
    test_labels = set(test_df[category_col].astype(str))
    only_in_test = test_labels - train_labels
    if only_in_test:
        raise ValueError(f"Unseen labels in train after dropping: {sorted(list(only_in_test))[:20]}")

    # cleanup helper column
    train_df = train_df.drop(columns=["donor_key"])
    test_df = test_df.drop(columns=["donor_key"])

    print(f"[Split] train={len(train_df)}, test={len(test_df)}, test_frac={len(test_df)/(len(train_df)+len(test_df)):.3f}")

    return train_df, test_df, test_donor_keys

def donor_wise_split_with_balancing(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    match_keys: List[str],
    study_col: str = "dataset_id",
    donor_col: str = "donor_id",
    test_fraction_by_samples: float = 0.20,
    hard_cap_fraction_by_samples: float = 0.40,
    random_state: int = 2025,
    min_unique_donor_keys_for_split: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:

    if isinstance(match_keys, str):
        match_keys = [match_keys]
    else:
        match_keys = list(match_keys)

    for df_name, df in [("reference_df", reference_df), ("target_df", target_df)]:
        for col in [study_col, donor_col]:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in {df_name}.")
        missing = [k for k in match_keys if k not in df.columns]
        if missing:
            raise KeyError(f"Missing match_keys in {df_name}: {missing}")

    if test_fraction_by_samples <= 0 or test_fraction_by_samples >= 1:
        raise ValueError("test_fraction_by_samples must be in (0, 1).")
    if hard_cap_fraction_by_samples <= 0 or hard_cap_fraction_by_samples >= 1:
        raise ValueError("hard_cap_fraction_by_samples must be in (0, 1).")
    if hard_cap_fraction_by_samples < test_fraction_by_samples:
        raise ValueError("hard_cap_fraction_by_samples must be >= test_fraction_by_samples.")

    ref = reference_df.copy()
    tgt = target_df.copy()

    for df in [ref, tgt]:
        df[study_col] = df[study_col].astype(str).str.strip()
        df[donor_col] = df[donor_col].astype(str).str.strip()
        df = df[df[donor_col].notna() & (df[donor_col].astype(str).str.strip() != "")]
    ref = ref[ref[donor_col].notna() & (ref[donor_col].astype(str).str.strip() != "")]
    tgt = tgt[tgt[donor_col].notna() & (tgt[donor_col].astype(str).str.strip() != "")]
    if ref.empty or tgt.empty:
        raise ValueError(f"Empty after filtering invalid donor_id: ref={len(ref)}, tgt={len(tgt)}.")

    ref["donor_key"] = ref[study_col] + "||" + ref[donor_col]
    tgt["donor_key"] = tgt[study_col] + "||" + tgt[donor_col]

    if ref["donor_key"].nunique() < min_unique_donor_keys_for_split:
        raise ValueError(
            f"reference_df has only {ref['donor_key'].nunique()} unique donor_key "
            f"(need >= {min_unique_donor_keys_for_split})."
        )
    if tgt["donor_key"].nunique() < min_unique_donor_keys_for_split:
        raise ValueError(
            f"target_df has only {tgt['donor_key'].nunique()} unique donor_key "
            f"(need >= {min_unique_donor_keys_for_split})."
        )

    def _pick_test_donors(df: pd.DataFrame, side_name: str) -> List[str]:
        donor_counts = df["donor_key"].value_counts()
        donor_keys_all = donor_counts.index.tolist()

        total_n = int(len(df))
        target_min = int(math.floor(total_n * float(test_fraction_by_samples)))
        hard_max = int(math.floor(total_n * float(hard_cap_fraction_by_samples)))

        if target_min <= 0:
            raise ValueError(f"{side_name}: test_fraction_by_samples too small; target test sample count is 0.")
        if hard_max <= 0:
            raise ValueError(f"{side_name}: hard_cap_fraction_by_samples too small; hard cap test sample count is 0.")
        if hard_max >= total_n:
            raise ValueError(f"{side_name}: hard_cap_fraction_by_samples must keep hard cap < total samples.")

        rng = np.random.RandomState(random_state)
        donor_keys = donor_keys_all.copy()
        rng.shuffle(donor_keys)
        donor_keys_small_to_large = sorted(donor_keys, key=lambda k: int(donor_counts.loc[k]))

        test_keys: List[str] = []
        test_n = 0

        for k in donor_keys_small_to_large:
            c = int(donor_counts.loc[k])
            if test_n + c <= target_min:
                test_keys.append(k)
                test_n += c
            else:
                break

        remaining = [k for k in donor_keys_small_to_large if k not in test_keys]
        if test_n < target_min and remaining:
            k_extra = remaining[0]
            c_extra = int(donor_counts.loc[k_extra])
            if test_n + c_extra <= hard_max:
                test_keys.append(k_extra)
                test_n += c_extra

        if not test_keys:
            test_keys = [donor_keys_small_to_large[0]]

        train_keys_left = set(donor_keys_all) - set(test_keys)
        if not train_keys_left:
            raise ValueError(f"{side_name}: split would leave empty train; cannot split.")

        return test_keys

    ref_test_donor_keys = _pick_test_donors(ref, "reference")
    tgt_test_donor_keys = _pick_test_donors(tgt, "target")

    test_donor_keys = sorted(set(ref_test_donor_keys) | set(tgt_test_donor_keys))

    ref_test = ref[ref["donor_key"].isin(set(test_donor_keys))].copy()
    ref_train = ref[~ref["donor_key"].isin(set(test_donor_keys))].copy()
    tgt_test = tgt[tgt["donor_key"].isin(set(test_donor_keys))].copy()
    tgt_train = tgt[~tgt["donor_key"].isin(set(test_donor_keys))].copy()

    if ref_train.empty or ref_test.empty or tgt_train.empty or tgt_test.empty:
        raise ValueError(
            "Donor split produced an empty side. "
            f"ref_train={len(ref_train)} ref_test={len(ref_test)} "
            f"tgt_train={len(tgt_train)} tgt_test={len(tgt_test)}."
        )

    ref_train = ref_train.drop(columns=["donor_key"])
    ref_test = ref_test.drop(columns=["donor_key"])
    tgt_train = tgt_train.drop(columns=["donor_key"])
    tgt_test = tgt_test.drop(columns=["donor_key"])

    train_balanced, test_balanced = balance_for_training(
        ref_train=ref_train,
        ref_test=ref_test,
        tgt_train=tgt_train,
        tgt_test=tgt_test,
        match_keys=match_keys,
        study_col=study_col,
        donor_col=donor_col,
        random_state=random_state,
    )

    return train_balanced, test_balanced, tgt_train, tgt_test, test_donor_keys