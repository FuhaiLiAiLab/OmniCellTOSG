# data_loader.py
import numpy as np
import os
import re

DATASET_PATH_DICT = {
    "adrenal_gland": {
        "general": "{root}/adrenal_gland/general",
        "lung_adenocarcinoma": "{root}/adrenal_gland/lung_adenocarcinoma",
        "SCLC": "{root}/adrenal_gland/small_cell_lung_carcinoma",
    },
    "blood": {
        "ccRCC": "{root}/blood/clear_cell_renal_carcinoma",
        "CIS": "{root}/blood/clinically_isolated_syndrome",
        "general": "{root}/blood/general",
        "nasopharyngeal_carcinoma": "{root}/blood/nasopharyngeal_carcinoma",
        "MS": "{root}/blood/multiple_sclerosis",
    },
    "bone_marrow": {
        "AML": "{root}/bone_marrow/acute_myeloid_leukemia",
        "APL": "{root}/bone_marrow/acute_promyelocytic_leukemia",
        "NHL": "{root}/bone_marrow/b-cell_non-hodgkin_lymphoma",
        "general": "{root}/bone_marrow/general",
        "prostate_cancer_bone_metastases": "{root}/bone_marrow/prostate_cancer_bone_metastases",
        "refractory_multiple_myeloma": "{root}/bone_marrow/refractory_multiple_myeloma",
    },
    "brain": {
        "AD": "{root}/brain/alzheimer's_disease",
        "ASL": "{root}/brain/amyotrophic_lateral_sclerosis",
        "anaplastic_astrocytoma": "{root}/brain/anaplastic_astrocytoma",
        "autism_spectrum_disorder": "{root}/brain/autism_spectrum_disorder",
        "diffuse_intrinsic_pontine_glioma": "{root}/brain/diffuse_intrinsic_pontine_glioma",
        "epilepsy": "{root}/brain/epilepsy",
        "frontotemporal_dementia": "{root}/brain/frontotemporal_dementia",
        "general": "{root}/brain/general",
        "glioblastoma": "{root}/brain/glioblastoma",
        "gliomas": "{root}/brain/gliomas",
        "lung_adenocarcinoma": "{root}/brain/lung_adenocarcinoma",
        "major_depressive_disorder": "{root}/brain/major_depressive_disorder",
        "melanoma_brain_metastases": "{root}/brain/melanoma_brain_metastases",
        "MS": "{root}/brain/multiple_sclerosis",
        "oligodendroglioma": "{root}/brain/oligodendroglioma",
        "pilocytic_astrocytoma": "{root}/brain/pilocytic_astrocytoma",
    },
    "breast": {
        "general": "{root}/breast/general",
    },
    "cervical_spinal_cord": {
        "general": "{root}/cervical_spinal_cord/general",
    },
    "esophagus": {
        "general": "{root}/esophagus/general",
    },
    "eye": {
        "general": "{root}/eye/general",
    },
    "gonad": {
        "general": "{root}/gonad/general",
    },
    "heart": {
        "general": "{root}/heart/general",
    },
    "intestine": {
        "adenocarcinoma": "{root}/intestine/adenocarcinoma",
        "general": "{root}/intestine/general",
        "neuroendocrine_carcinoma": "{root}/intestine/neuroendocrine_carcinoma",
    },
    "kidney": {
        "chromophobe_renal_cell_carcinoma": "{root}/kidney/chromophobe_renal_cell_carcinoma",
        "ccRCC": "{root}/kidney/clear_cell_renal_carcinoma",
        "general": "{root}/kidney/general",
        "wilms_tumor": "{root}/kidney/wilms_tumor",
    },
    "liver": {
        "blastoma": "{root}/liver/blastoma",
        "general": "{root}/liver/general",
        "intrahepatic_cholangiocarcinoma": "{root}/liver/intrahepatic_cholangiocarcinoma",
        "lung_adenocarcinoma": "{root}/liver/lung_adenocarcinoma",
        "SCLC": "{root}/liver/small_cell_lung_carcinoma",
    },
    "lung": {
        "general": "{root}/lung/general",
        "lung_adenocarcinoma": "{root}/lung/lung_adenocarcinoma",
        "non_small_cell_lung_carcinoma": "{root}/lung/non-small_cell_lung_carcinoma",
        "SCLC": "{root}/lung/small_cell_lung_carcinoma",
        "squamous_cell_lung_carcinoma": "{root}/lung/squamous_cell_lung_carcinoma",
    },
    "lymph_node": {
        "follicular_lymphoma": "{root}/lymph_node/follicular_lymphoma",
        "general": "{root}/lymph_node/general",
        "lung_adenocarcinoma": "{root}/lymph_node/lung_adenocarcinoma",
        "SCLC": "{root}/lymph_node/small_cell_lung_carcinoma",
        "squamous_cell_lung_carcinoma": "{root}/lymph_node/squamous_cell_lung_carcinoma",
    },
    "mouth": {
        "general": "{root}/mouth/general",
    },
    "nasopharynx": {
        "nasopharyngeal_carcinoma": "{root}/nasopharynx/nasopharyngeal_carcinoma",
    },
    "pancreas": {
        "general": "{root}/pancreas/general",
    },
    "skin": {
        "acne": "{root}/skin/acne",
        "CTCL": "{root}/skin/advanced-stage_cutaneous_t-cell_lymphoma",
        "general": "{root}/skin/general",
        "melanoma_peripheral_metastases": "{root}/skin/melanoma_peripheral_metastases",
    },
    "stomach": {
        "gastric_cancer": "{root}/stomach/gastric_cancer",
        "general": "{root}/stomach/general",
    },
    "uterus": {
        "general": "{root}/uterus/general",
    },
}

def load_data_from_dict(root, organ=None, disease=None, label_type="ct", seed=2025, ratio=0.01, train_text=False, train_bio=False, shuffle=False):
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
        print(f"\nChecking dataset path: {dataset_path}")
        all_files = os.listdir(dataset_path)
        print(f"Files in dataset path: {all_files}")

        # Extract and sort based on partition number
        X_files = sorted(
            [f for f in all_files if "_X_partition_" in f and f.endswith(".npy")],
            key=lambda x: int(re.search(r"partition_(\d+)", x).group(1))
        )
        Y_files = sorted(
            [f for f in all_files if "_Y_partition_" in f and f.endswith(".npy")],
            key=lambda x: int(re.search(r"partition_(\d+)", x).group(1))
        )

        # Debugging print to check sorted lists
        print(f"\nSorted X files: {X_files}")
        print(f"Sorted Y files: {Y_files}")

        # Ensure pairing of X and Y files
        for X_file, Y_file in zip(X_files, Y_files):
            print(f"\nPairing: {X_file} ↔ {Y_file}")

            X_data = _load_npy(os.path.join(dataset_path, X_file))
            Y_data = _load_npy(os.path.join(dataset_path, Y_file))[:, label_index]

            sample_size = max(1, int(X_data.shape[0] * ratio))
            indices = np.random.choice(X_data.shape[0], sample_size, replace=False)

            print(f"Sampling {sample_size} rows from {X_file}")

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

    # Load name and description embeddings or raw text based on train_text flag
    if train_text:
        s_name = _load_csv(os.path.join(root, "s_name.csv"))
        s_desc = _load_csv(os.path.join(root, "s_desc.csv"))
    else:
        s_name = _load_npy(os.path.join(root, "x_name_emb.npy"))
        s_desc = _load_npy(os.path.join(root, "x_desc_emb.npy"))

    # Load biological embeddings or raw bio data based on train_bio flag
    if train_bio:
        s_bio = _load_csv(os.path.join(root, "s_bio.csv"))
    else:
        s_bio = _load_npy(os.path.join(root, "x_bio_emb.npy"))

    print(f"\nFinal dataset shape: X {xAll.shape}, Y {yAll.shape}")
    return xAll, yAll, edge_index, internal_edge_index, ppi_edge_index, s_name, s_desc, s_bio

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