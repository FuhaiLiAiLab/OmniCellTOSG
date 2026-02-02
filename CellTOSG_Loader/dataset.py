import os
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

from .subset_builder import CellTOSGSubsetBuilder


class CellTOSGDataLoader:
    LABEL_ZERO_LABELS_BY_LABEL_COLUMN: dict[str, set[str]] = {
        "sex": {"female"},
        "disease": {"normal", "unclassified", "unknown"},
        "cell_type": {"unknown", "unclassified"},
    }

    def __init__(
        self,
        root,
        conditions: dict,
        shuffle: bool = False,
        stratified_balancing: bool = False,
        task: Optional[str] = None,
        label_column: Optional[str] = None,
        sample_ratio: Optional[float] = None,
        sample_size: Optional[int] = None,
        train_text: bool = False,
        train_bio: bool = False,
        extract_mode: Optional[str] = None,
        random_state: int = 2025,
        correction_method=None, # None / "combat_seq"
        output_dir: Optional[str] = None,
    ):
        self.root = root
        self.extract_mode = extract_mode
        self.conditions = conditions
        self.task = task
        self.stratified_balancing = stratified_balancing
        self.shuffle = shuffle
        self.label_column = label_column
        self.sample_ratio = sample_ratio
        self.sample_size = sample_size
        self.train_text = train_text
        self.train_bio = train_bio
        self.random_state = random_state
        self.correction_method = correction_method
        self.output_dir = output_dir

        self.query = CellTOSGSubsetBuilder(root=self.root)
        print("Welcome to use CellTOSGDataset V2.1.0.")
        print(f"[CellTOSGDataset] Initialized with root: {self.root}")
        print("[CellTOSGDataset] Previewing sample distribution:")
        self.df_preview = self.query.view(self.conditions)

        if sample_ratio is not None and sample_size is not None:
            raise ValueError("Only one of sample_ratio or sample_size can be specified.")

        res = self.query.extract(
            extract_mode=self.extract_mode,
            task=self.task,
            stratified_balancing=self.stratified_balancing,
            shuffle=self.shuffle,
            sample_ratio=self.sample_ratio,
            sample_size=self.sample_size,
            random_state=self.random_state,
            correction_method=self.correction_method,
            output_dir=self.output_dir,
        )

        if isinstance(res, dict) and res.get("status") == "NO_SUBSET_RETRIEVED":
            print("[CellTOSGDataset] No subset retrieved. Available conditions:")
            print(res)
            raise ValueError("NO_SUBSET_RETRIEVED")

        self.label_mapping = None

        if self.extract_mode == "train":
            (train_x, train_df), (test_x, test_df) = res

            train_df = train_df.copy()
            test_df = test_df.copy()
            train_df["split"] = "train"
            test_df["split"] = "test"

            self.data = {"train": train_x, "test": test_x}
            self.metadata = {"train": train_df, "test": test_df}

            if self.label_column:
                y_train, y_test, mapping = self.build_split_labels(
                    train_df=train_df,
                    test_df=test_df,
                    label_column=self.label_column,
                    task=self.task,
                )
                self.labels = {"train": y_train, "test": y_test}
                self.label_mapping = mapping

                if self.output_dir:
                    self.save_split_labels_and_mapping(
                        train_df=train_df,
                        test_df=test_df,
                        y_train=y_train,
                        y_test=y_test,
                        mapping=mapping,
                        label_column=self.label_column,
                    )
            else:
                self.labels = {"train": train_df, "test": test_df}

        else:
            data, df = res
            self.data = data
            self.metadata = df

            # In inference mode, labels should stay as metadata df (no numeric mapping)
            self.labels = df

        self.edge_index = self._load_npy("edge_index.npy")
        self.internal_edge_index = self._load_npy("internal_edge_index.npy")
        self.ppi_edge_index = self._load_npy("ppi_edge_index.npy")

        if train_text:
            print("[CellTOSGDataset] Loading raw text data for training...")
            self.s_name = self._load_csv("s_name.csv")
            self.s_desc = self._load_csv("s_desc.csv")
        else:
            print("[CellTOSGDataset] Loading precomputed text embeddings...")
            self.x_name_emb = self._load_npy("x_name_emb.npy")
            self.x_desc_emb = self._load_npy("x_desc_emb.npy")

        if train_bio:
            print("[CellTOSGDataset] Loading raw biological data for training...")
            self.s_bio = self._load_csv("s_bio.csv")
        else:
            print("[CellTOSGDataset] Loading precomputed biological embeddings...")
            self.x_bio_emb = self._load_npy("x_bio_emb.npy")

    def build_split_labels(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label_column: str,
        task: str | None,
    ):
        resolved_label_col = self.query.FIELD_ALIAS.get(label_column, label_column)

        if resolved_label_col not in train_df.columns:
            raise ValueError(f"Label column '{resolved_label_col}' not found in train metadata.")
        if resolved_label_col not in test_df.columns:
            raise ValueError(f"Label column '{resolved_label_col}' not found in test metadata.")

        merged = pd.concat([train_df, test_df], ignore_index=True)

        # all unique labels (keep original strings)
        all_labels = merged[resolved_label_col].dropna().unique().tolist()

        # label_zero labels: collapse them all into class 0 if present
        label_zero_set = self.LABEL_ZERO_LABELS_BY_LABEL_COLUMN.get(label_column, set())
        label_zero_lower = {str(p).strip().lower() for p in label_zero_set}

        label_zero_present = []
        other_labels = []
        for lab in all_labels:
            lab_norm = str(lab).strip().lower()
            if lab_norm in label_zero_lower:
                label_zero_present.append(lab)
            else:
                other_labels.append(lab)

        other_labels_sorted = sorted(set(other_labels), key=lambda x: str(x).strip().lower())

        mapping: dict[str, int] = {}

        # Map all label_zero labels to 0
        for lab in set(label_zero_present):
            mapping[lab] = 0

        # Map remaining labels to 1..K
        start_idx = 1 if len(label_zero_present) > 0 else 0

        for idx, lab in enumerate(other_labels_sorted, start=start_idx):
            mapping[lab] = idx

        # Encode
        y_train = train_df[resolved_label_col].map(mapping)
        y_test = test_df[resolved_label_col].map(mapping)

        # If any label is missing in mapping (should not happen), mark as -1
        y_train = y_train.fillna(-1).astype(int).values
        y_test = y_test.fillna(-1).astype(int).values

        return y_train, y_test, mapping

    def save_split_labels_and_mapping(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        mapping: dict[str, int],
        label_column: str,
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        mapping_df = pd.DataFrame(
            {
                "label_name": list(mapping.keys()),
                "label_index": list(mapping.values()),
            }
        )
        mapping_path = os.path.join(self.output_dir, f"label_mapping_{label_column}.csv")
        mapping_df.to_csv(mapping_path, index=False)
        print(f"[Label Mapping] Saved to: {mapping_path}")

        train_out = train_df.copy()
        test_out = test_df.copy()
        train_out["label_index"] = y_train
        test_out["label_index"] = y_test

        full_path = os.path.join(self.output_dir, f"labels_full_{label_column}.csv")
        pd.concat([train_out, test_out], ignore_index=True).to_csv(full_path, index=False)
        print(f"[Label Full File] Saved to: {full_path}")

        np.save(os.path.join(self.output_dir, "train", "labels_train.npy"), y_train)
        np.save(os.path.join(self.output_dir, "test", "labels_test.npy"), y_test)
        print(f"[Label NPY] Saved encoded labels to: {os.path.join(self.output_dir, 'train', 'labels_train.npy')}")
        print(f"[Label NPY] Saved encoded labels to: {os.path.join(self.output_dir, 'test', 'labels_test.npy')}")

    def _load_npy(self, fname: str):
        path = os.path.join(self.root, fname)
        return np.load(path) if os.path.exists(path) else None

    def _load_csv(self, fname: str):
        path = os.path.join(self.root, fname)
        return pd.read_csv(path) if os.path.exists(path) else None
