import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Set, Optional

def balance_for_inference(
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

    # Stage definitions and unknown checker
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

    # Validate required columns
    if stage_col not in reference_df.columns or stage_col not in target_df.columns:
        raise KeyError(f"stage_col '{stage_col}' must be present in both DataFrames.")

    # Keep only rows where stage_col is not NA (unknown string is still valid)
    reference_df = reference_df[reference_df[stage_col].notna()].copy()
    target_df = target_df[target_df[stage_col].notna()].copy()

    # Locate column indices for keys
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

    # Group by match_keys
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
            # Strict matching
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

def balance_for_training(
    ref_train: pd.DataFrame,
    ref_test: pd.DataFrame,
    tgt_train: pd.DataFrame,
    tgt_test: pd.DataFrame,
    match_keys: List[str],
    study_col: str = "dataset_id",
    donor_col: str = "donor_id",
    stage_col: str = "development_stage_category",
    max_stage_offset: int = 2,
    upsample: bool = True,
    random_state: int = 2025,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if isinstance(match_keys, str):
        match_keys = [match_keys]
    else:
        match_keys = list(match_keys)

    required_cols = set(match_keys) | {study_col, donor_col, stage_col}

    for df_name, df in [
        ("ref_train", ref_train),
        ("ref_test", ref_test),
        ("tgt_train", tgt_train),
        ("tgt_test", tgt_test),
    ]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in {df_name}: {missing}")

    if ref_test.empty or tgt_test.empty:
        raise ValueError(
            f"Cannot balance test split: ref_test={len(ref_test)}, tgt_test={len(tgt_test)}. "
            "Donor split must keep both classes in test."
        )
    if ref_train.empty or tgt_train.empty:
        raise ValueError(
            f"Cannot balance train split: ref_train={len(ref_train)}, tgt_train={len(tgt_train)}. "
            "Donor split must keep both classes in train."
        )

    def _make_donor_key(df: pd.DataFrame) -> pd.Series:
        return (
            df[study_col].astype(str).str.strip()
            + "||"
            + df[donor_col].astype(str).str.strip()
        )

    def _align_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            if c not in out.columns:
                out[c] = pd.NA
        return out.loc[:, cols]

    ref_cols = list(ref_train.columns)

    # print(f"[Debug] ref_train size={len(ref_train)} ref_test size={len(ref_test)}")
    # print(f"[Debug] tgt_train size={len(tgt_train)} tgt_test size={len(tgt_test)}")

    # print(f"[Debug] ref_train unique donor_key={_make_donor_key(ref_train).nunique()}")
    # print(f"[Debug] ref_test unique donor_key={_make_donor_key(ref_test).nunique()}")
    # print(f"[Debug] tgt_train unique donor_key={_make_donor_key(tgt_train).nunique()}")
    # print(f"[Debug] tgt_test unique donor_key={_make_donor_key(tgt_test).nunique()}")

    test_matched_target, test_matched_keys = balance_for_inference(
        reference_df=ref_test,
        target_df=tgt_test,
        match_keys=match_keys,
        stage_col=stage_col,
        max_stage_offset=max_stage_offset,
        upsample=upsample,
        random_state=random_state,
    )

    ref_test_keep = ref_test[
        ref_test[match_keys].apply(tuple, axis=1).isin(test_matched_keys)
    ].copy()

    test_matched_target = _align_columns(test_matched_target, ref_cols)
    ref_test_keep = _align_columns(ref_test_keep, ref_cols)
    test_balanced = pd.concat([ref_test_keep, test_matched_target], ignore_index=True)

    # print(f"[Debug] test_matched_keys size={len(test_matched_keys)}")
    # print(f"[Debug] test_matched_target size={len(test_matched_target)}")
    # print(f"[Debug] ref_test_keep size={len(ref_test_keep)}")
    # print(f"[Debug] test_balanced size={len(test_balanced)}")

    train_matched_target, train_matched_keys = balance_for_inference(
        reference_df=ref_train,
        target_df=tgt_train,
        match_keys=match_keys,
        stage_col=stage_col,
        max_stage_offset=max_stage_offset,
        upsample=upsample,
        random_state=random_state,
    )

    ref_train_keep = ref_train[
        ref_train[match_keys].apply(tuple, axis=1).isin(train_matched_keys)
    ].copy()

    train_matched_target = _align_columns(train_matched_target, ref_cols)
    ref_train_keep = _align_columns(ref_train_keep, ref_cols)
    train_balanced = pd.concat([ref_train_keep, train_matched_target], ignore_index=True)

    # print(f"[Debug] train_matched_keys size={len(train_matched_keys)}")
    # print(f"[Debug] train_matched_target size={len(train_matched_target)}")
    # print(f"[Debug] ref_train_keep size={len(ref_train_keep)}")
    # print(f"[Debug] train_balanced size={len(train_balanced)}")

    if train_balanced.empty:
        raise ValueError("train_balanced is empty after balancing; no matched keys in tgt_train.")
    if test_balanced.empty:
        raise ValueError(
            "test_balanced is empty after balancing; no matched keys in tgt_test. "
            "Match keys may be too strict or test split does not cover needed (cell, sex, stage)."
        )

    train_donors = set(_make_donor_key(train_balanced))
    test_donors = set(_make_donor_key(test_balanced))
    overlap = train_donors & test_donors
    if overlap:
        raise ValueError(f"Leakage: train/test share donor_key (examples): {sorted(list(overlap))[:10]}")

    return train_balanced, test_balanced