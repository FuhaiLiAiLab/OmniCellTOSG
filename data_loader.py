# data_loader.py
import numpy as np
import os
import re

DATASET_PATH_DICT = {
    "brain": {
        "AD": "{root}/brain/alzheimer's_disease",
        "general": "{root}/brain/general"
    },
    "bone_marrow": {
        "acute_myeloid_leukemia": "{root}/bone_marrow/acute_myeloid_leukemia",
        "acute_lymphoblastic_leukemia": "{root}/bone_marrow/acute_lymphoblastic_leukemia",
        "general": "{root}/bone_marrow/general"
    },
    "lung": {
        "SCLC": "{root}/lung/small_cell_lung_carcinoma",
        "general": "{root}/lung/general"
    },
    "kidney": {
        "RCC": "{root}/kidney/clear_renal_cell_carcinoma",
        "general": "{root}/kidney/general"
    }
}

def load_data_from_dict(root, organ=None, disease=None, label_type="ct", seed=2025, ratio=0.01, shuffle=False):
    """
    Load .npy and additional data files from the root directory.
    For X and Y data:
    - Reads all npy files in the directory in partition order.
    - Samples 1% of each file using a fixed seed.
    - Concatenates them along the row (axis=0).
    - Optionally shuffles the combined data and labels.
    """
    np.random.seed(seed)
    label_index = {"ct": 0, "og": 1, "ds": 2, "status": 3}[label_type]

    dataset_paths = []

    # If only organ is provided: load all diseases for that organ
    if organ is not None and disease is None:
        organ_diseases = DATASET_PATH_DICT.get(organ, {})
        if not organ_diseases:
            raise ValueError(f"No data found for organ: {organ}")
        for _, path_template in organ_diseases.items():
            dataset_paths.append(path_template.format(root=root))

    # If only disease is provided: load it across all available organs
    elif organ is None and disease is not None:
        for current_organ, diseases in DATASET_PATH_DICT.items():
            if disease in diseases:
                dataset_paths.append(diseases[disease].format(root=root))

    # If both organ and disease are provided: load the specific pair
    else:
        dataset_path_template = DATASET_PATH_DICT.get(organ, {}).get(disease, None)
        if dataset_path_template is None:
            raise ValueError(f"Invalid organ/disease: {organ}/{disease}")
        dataset_paths.append(dataset_path_template.format(root=root))


    # Load and combine data from all matched paths
    X_list, Y_list = [], []
    total_disease_samples = 0

    for dataset_path in dataset_paths:
        print(f"Files in dataset path: {os.listdir(dataset_path)}")
        X_files = sorted(
            [f for f in os.listdir(dataset_path) if "_X_partition_" in f and f.endswith(".npy")],
            key=lambda x: int(re.search(r"partition_(\d+)", x).group(1))
        )
        Y_files = sorted(
            [f for f in os.listdir(dataset_path) if "_Y_partition_" in f and f.endswith(".npy")],
            key=lambda x: int(re.search(r"partition_(\d+)", x).group(1))
        )

        for X_file, Y_file in zip(X_files, Y_files):
            X_data = _load_npy(os.path.join(dataset_path, X_file))
            Y_data = _load_npy(os.path.join(dataset_path, Y_file))[:, label_index]

            sample_size = max(1, int(X_data.shape[0] * ratio))
            indices = np.random.choice(X_data.shape[0], sample_size, replace=False)

            X_list.append(X_data[indices])
            Y_list.append(Y_data[indices])
            total_disease_samples += sample_size

    xAll = np.concatenate(X_list, axis=0)
    yAll = np.concatenate(Y_list, axis=0)

    # If status label and not general disease, balance with general samples
    if label_type == "status" and disease != "general":
        print(f"Balancing with general samples for organ: {organ} (Need {total_disease_samples} samples)")
        general_path_template = DATASET_PATH_DICT.get(organ, {}).get("general", None)
        if general_path_template is None:
            raise ValueError(f"No general data available for organ: {organ}")

        general_dataset_path = general_path_template.format(root=root)
        general_X_files = sorted(
            [f for f in os.listdir(general_dataset_path) if "_X_partition_" in f and f.endswith(".npy")],
            key=lambda x: int(re.search(r"partition_(\d+)", x).group(1))
        )
        general_Y_files = sorted(
            [f for f in os.listdir(general_dataset_path) if "_Y_partition_" in f and f.endswith(".npy")],
            key=lambda x: int(re.search(r"partition_(\d+)", x).group(1))
        )

        general_samples_collected = 0
        general_X_list, general_Y_list = [], []

        # Sample general data until it matches total_disease_samples
        for X_file, Y_file in zip(general_X_files, general_Y_files):
            if general_samples_collected >= total_disease_samples:
                break
            X_data = _load_npy(os.path.join(general_dataset_path, X_file))
            Y_data = _load_npy(os.path.join(general_dataset_path, Y_file))[:, label_index]

            sample_size = min(total_disease_samples - general_samples_collected,
                              max(1, int(X_data.shape[0] * ratio * 2)))
            indices = np.random.choice(X_data.shape[0], sample_size, replace=False)

            general_X_list.append(X_data[indices])
            general_Y_list.append(Y_data[indices])
            general_samples_collected += sample_size

        # Combine disease and general data
        xAll = np.concatenate([xAll] + general_X_list, axis=0)
        yAll = np.concatenate([yAll] + general_Y_list, axis=0)

    # Shuffle combined data if requested
    if shuffle:
        xAll, yAll = _shuffle_data(xAll, yAll, seed)
    # Load shared edge indices and descriptions
    edge_index = _load_npy(os.path.join(root, "edge_index.npy"))
    internal_edge_index = _load_npy(os.path.join(root, "internal_edge_index.npy"))
    ppi_edge_index = _load_npy(os.path.join(root, "ppi_edge_index.npy"))
    s_name = _load_csv(os.path.join(root, "s_name.csv"))
    s_desc = _load_csv(os.path.join(root, "s_desc.csv"))

    return xAll, yAll, edge_index, internal_edge_index, ppi_edge_index, s_name, s_desc

def _shuffle_data(X, Y, seed=2025):
    """
    Shuffles data and labels together while keeping them aligned.
    """
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    print("Data shuffled.")
    return X[indices], Y[indices]

def _load_npy(file_path):
    """
    Loads numpy array from the specified file path.
    """
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def _load_csv(file_path):
    """
    Loads text data from a specified file path.
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return file.read().strip()
    return ""