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
    target_df: pd.DataFrame,
    match_keys: List[str],
    study_col: str = "dataset_id",
    stage_col: str = "development_stage_category",
    max_stage_offset: int = 2,
    upsample: bool = True,
    random_state: int = 2025,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    for df_name, df in [("ref_train", ref_train), ("ref_test", ref_test), ("target_df", target_df)]:
        if study_col not in df.columns:
            raise KeyError(f"Column '{study_col}' not found in {df_name}.")
    if stage_col not in target_df.columns:
        raise KeyError(f"Column '{stage_col}' must exist in target_df.")

    # decide cell_key
    cell_key: Optional[str] = None
    if "CMT_id" in match_keys and "CMT_id" in target_df.columns:
        cell_key = "CMT_id"
    elif "CMT_name" in match_keys and "CMT_name" in target_df.columns:
        cell_key = "CMT_name"
    else:
        raise ValueError("Neither 'CMT_id' nor 'CMT_name' available in target_df for matching.")

    def _unmatched_keys(ref_df: pd.DataFrame, matched_keys: Set[Tuple]) -> Set[Tuple]:
        ref_all = set(ref_df[match_keys].apply(tuple, axis=1))
        return ref_all - matched_keys

    # global study order by size (desc)
    all_studies_desc = (
        target_df[study_col].astype(str).value_counts().sort_values(ascending=False).index.tolist()
    )

    # identify 'exclusive-source' studies (study hosts at least one cell type exclusively)
    ck_study_map = (
        target_df[[cell_key, study_col]]
        .dropna(subset=[cell_key, study_col])
        .astype({cell_key: str, study_col: str})
        .drop_duplicates()
        .groupby(cell_key)[study_col]
        .apply(set)
    )
    exclusive_celltypes = {ck for ck, sset in ck_study_map.items() if len(sset) == 1}
    exclusive_studies: Set[str] = set()
    for ck in exclusive_celltypes:
        exclusive_studies |= ck_study_map[ck]

    # TEST: initial study pool = ref_test's studies minus exclusive studies
    ref_test_studies: Set[str] = set(ref_test[study_col].astype(str).unique())
    test_study_pool: Set[str] = set(s for s in ref_test_studies if s not in exclusive_studies)

    # forbid using ref_test studies in target_train later (no leakage)
    forbid_for_train: Set[str] = set(ref_test_studies)

    # initial test pool rows
    tgt_test_pool = target_df[target_df[study_col].astype(str).isin(test_study_pool)].copy()

    test_matched_target, test_matched_keys = balance_for_inference(
        reference_df=ref_test,
        target_df=tgt_test_pool,
        match_keys=match_keys,
        stage_col=stage_col,
        max_stage_offset=max_stage_offset,
        upsample=upsample,
        random_state=random_state,
    )
    ref_test_keep = ref_test[ref_test[match_keys].apply(tuple, axis=1).isin(test_matched_keys)].copy()
    test_balanced = pd.concat([ref_test_keep, test_matched_target], ignore_index=True)

    # try to augment test pool with additional studies (not in ref_test, not exclusive)
    needed_keys = _unmatched_keys(ref_test, test_matched_keys)
    augmentation_candidates = [s for s in all_studies_desc if (s not in ref_test_studies and s not in exclusive_studies)]
    added_to_test: Set[str] = set()

    # quick feasibility check for adding a study
    STAGES = [
        "80 and over", "aged", "middle aged", "adult", "young adult", "adolescent",
        "child", "preschool child", "infant", "newborn", "fetal", "embryonic", "unknown",
    ]
    stage_order = {s: i for i, s in enumerate(STAGES)}

    def _can_help(study_id: str, missing_keys: Set[Tuple]) -> bool:
        sub = target_df[target_df[study_col].astype(str) == study_id]
        if sub.empty:
            return False
        if cell_key not in sub.columns or stage_col not in sub.columns:
            return False
        sub_cells = set(sub[cell_key].astype(str))
        sub_stage_vals = set(sub[stage_col].astype(str).map(str.lower))

        for key in missing_keys:
            # locate indices for cell and stage
            try:
                idx_cell = match_keys.index(cell_key)
            except ValueError:
                idx_cell = None
            try:
                idx_stage = match_keys.index(stage_col)
            except ValueError:
                idx_stage = None

            cell_val = str(key[idx_cell]) if idx_cell is not None else None
            stage_val = str(key[idx_stage]).lower() if idx_stage is not None else None

            if (cell_val is None) or (cell_val not in sub_cells):
                continue
            if stage_val is None:
                return True
            s_idx = stage_order.get(stage_val, None)
            if s_idx is None:
                return True
            # any stage within backoff window
            for off in range(0, max_stage_offset + 1):
                alt_idx = s_idx + off
                if 0 <= alt_idx < len(STAGES) and STAGES[alt_idx] in sub_stage_vals:
                    return True
        return False

    improved = True
    while needed_keys and augmentation_candidates and improved:
        improved = False
        for st in list(augmentation_candidates):
            if not _can_help(st, needed_keys):
                continue
            # add the study to test pool
            add_rows = target_df[target_df[study_col].astype(str) == st]
            if add_rows.empty:
                augmentation_candidates.remove(st)
                continue
            tgt_test_pool = pd.concat([tgt_test_pool, add_rows], ignore_index=True)
            test_study_pool.add(st)
            added_to_test.add(st)
            augmentation_candidates.remove(st)

            # re-run matching
            test_matched_target, test_matched_keys = balance_for_inference(
                reference_df=ref_test,
                target_df=tgt_test_pool,
                match_keys=match_keys,
                stage_col=stage_col,
                max_stage_offset=max_stage_offset,
                upsample=upsample,
                random_state=random_state,
            )
            needed_after = _unmatched_keys(ref_test, test_matched_keys)

            if len(needed_after) < len(needed_keys):
                needed_keys = needed_after
                ref_test_keep = ref_test[ref_test[match_keys].apply(tuple, axis=1).isin(test_matched_keys)].copy()
                test_balanced = pd.concat([ref_test_keep, test_matched_target], ignore_index=True)
                improved = True

    # Train: build pool from remaining studies
    # train cant use： study in ref_test_studies + added_to_test
    forbidden_for_train = forbid_for_train.union(test_study_pool)
    tgt_train_pool = target_df[~target_df[study_col].astype(str).isin(forbidden_for_train)].copy()

    # run train matching
    train_matched_target, train_matched_keys = balance_for_inference(
        reference_df=ref_train,
        target_df=tgt_train_pool,
        match_keys=match_keys,
        stage_col=stage_col,
        max_stage_offset=max_stage_offset,
        upsample=upsample,
        random_state=random_state,
    )
    ref_train_keep = ref_train[ref_train[match_keys].apply(tuple, axis=1).isin(train_matched_keys)].copy()
    train_balanced = pd.concat([ref_train_keep, train_matched_target], ignore_index=True)

    return train_balanced, test_balanced
