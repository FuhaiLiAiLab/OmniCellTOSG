import os
import numpy as np
import pandas as pd
from .subset_builder import CellTOSGSubsetBuilder

class CellTOSGDataLoader:
    def __init__(
        self,
        root,
        conditions: dict,
        shuffle=False,
        balanced=False,
        downstream_task=None,
        label_column=None,
        sample_ratio=None,
        sample_size=None,
        random_state=42,
        output_dir=None
    ):
        self.root = root
        self.conditions = conditions
        self.shuffle = shuffle
        self.balanced = balanced
        self.downstream_task = downstream_task
        self.label_column = label_column
        self.sample_ratio = sample_ratio
        self.sample_size = sample_size
        self.random_state = random_state
        self.output_dir = output_dir

        self.query = CellTOSGSubsetBuilder(root=self.root)
        print("[CellTOSGDataset] Previewing sample distribution:")
        self.df_preview = self.query.view(self.conditions)

        if sample_ratio is not None and sample_size is not None:
            raise ValueError("Only one of sample_ratio or sample_size can be specified.")

        self.data, df = self.query.extract(
            shuffle=self.shuffle,
            balanced=self.balanced,
            downstream_task=self.downstream_task,
            sample_ratio=self.sample_ratio,
            sample_size=self.sample_size,
            random_state=self.random_state,
            output_dir=None
        )

        self.metadata = df

        if self.label_column:
            resolved_label_col = self.query.FIELD_ALIAS.get(self.label_column, self.label_column)
            if resolved_label_col not in df.columns:
                raise ValueError(f"Label column '{resolved_label_col}' not found in metadata.")
            
            priority_labels = {"female", "normal", "healthy"}

            # Clean and extract unique labels
            all_labels = df[resolved_label_col].dropna().unique().tolist()

            # Sort labels: priority ones first, others follow alphabetically
            sorted_labels = sorted(set(all_labels), key=lambda x: (x not in priority_labels, x))

            # Build label mapping: priority label(s) → 0, rest increment from 1
            self.label_mapping = {}
            current_index = 0
            priority_assigned = False

            for label in sorted_labels:
                if label in priority_labels and not priority_assigned:
                    self.label_mapping[label] = 0
                    priority_assigned = True
                else:
                    if priority_assigned:
                        current_index = 1
                    self.label_mapping[label] = current_index
                    current_index += 1

            self.labels = df[resolved_label_col].map(self.label_mapping).astype(int).values

            os.makedirs(self.output_dir, exist_ok=True)

            mapping_df = pd.DataFrame({
                "label_name": list(self.label_mapping.keys()),
                "label_index": list(self.label_mapping.values())
            })
            mapping_path = os.path.join(self.output_dir, f"label_mapping_{self.label_column}.csv")
            mapping_df.to_csv(mapping_path, index=False)

            full_labels_path = os.path.join(self.output_dir, f"labels_full_{self.label_column}.csv")
            df_with_label = df.copy()
            df_with_label["label_index"] = self.labels
            df_with_label.to_csv(full_labels_path, index=False)

            print(f"[Label Mapping] Saved to: {mapping_path}")
            print(f"[Label Full File] Saved to: {full_labels_path}")
        else:
            self.labels = df

        self.edge_index = self._load_npy("edge_index.npy")
        self.internal_edge_index = self._load_npy("internal_edge_index.npy")
        self.ppi_edge_index = self._load_npy("ppi_edge_index.npy")
    
        self.s_name = self._load_npy("x_name_emb.npy")
        self.s_desc = self._load_npy("x_desc_emb.npy")
        self.s_bio = self._load_npy("x_bio_emb.npy")

    def _load_npy(self, fname):
        path = os.path.join(self.root, fname)
        return np.load(path) if os.path.exists(path) else None

    def _load_csv(self, fname):
        path = os.path.join(self.root, fname)
        return pd.read_csv(path) if os.path.exists(path) else None
