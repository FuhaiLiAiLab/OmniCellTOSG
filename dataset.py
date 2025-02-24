# dataset.py
from .data_loader import load_data_from_dict

import numpy as np


class CellTOSGDataset:
    """
    Load the data from the dictionary and return multiple components including
    data, labels, edge indices, and additional descriptions.
    """
    def __init__(self, root, categories, name, label_type="ct", seed=2025, ratio=0.01, shuffle=False):
        self.root = root
        self.seed = seed
        self.ratio = ratio
        organ, disease = None, None
        if categories == "get_organ":
            organ = name
        elif categories == "get_disease":
            disease = name
        elif categories == "get_organ_disease":
            organ, disease = name.split("-")

        # Load all data components with configurable seed and ratio
        (self.data, self.labels, self.edge_index,
         self.internal_edge_index, self.ppi_edge_index,
         self.s_name, self.s_desc) = load_data_from_dict(
            self.root, organ, disease, label_type, self.seed, self.ratio, shuffle
        )

    def __getitem__(self, idx):
        """
        Return a single sample containing all components.
        """
        return (self.data[idx], self.labels[idx], self.edge_index,
                self.internal_edge_index, self.ppi_edge_index,
                self.s_name, self.s_desc)

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.data)