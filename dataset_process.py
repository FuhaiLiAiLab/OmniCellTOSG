import os
import pandas as pd
import numpy as np
import anndata as ad
import gene_mapping

# Define partition size limit in MB
PARTITION_SIZE_MB = 1024 

def get_partitioned_files(disease_path, partition_size_mb):
    """
    Partitions h5ad files into groups where each partition contains files up to the specified size.

    Parameters:
    - disease_path: Path to the disease directory containing .h5ad files.
    - partition_size_mb: Maximum size of each partition in MB.

    Returns:
    - List of partitions, where each partition is a list of file paths.
    """
    h5ad_files = sorted([f for f in os.listdir(disease_path) if f.endswith(".h5ad")])
    partitions = []
    current_partition = []
    current_size = 0

    for file in h5ad_files:
        file_path = os.path.join(disease_path, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert bytes to MB

        if current_size + file_size > partition_size_mb:
            if current_partition:
                partitions.append(current_partition)
                current_partition = []
                current_size = 0

        current_partition.append(file_path)
        current_size += file_size

    if current_partition:
        partitions.append(current_partition)

    return partitions

# def encode_labels(grouped_file, value_list):
#     """
#     Generic encoding function for cell types, organs, and diseases.
#     This version handles case-insensitive matching by converting all values to lowercase.
#     """
#     grouped_data = pd.read_csv(grouped_file)
#     mapping = {}
#     for _, row in grouped_data.iterrows():
#         label = row["cluster"]
#         values = eval(row["group_values"])
#         for v in values:
#             mapping[v.lower()] = label  # Convert to lowercase for case-insensitive
    
#     unmapped = set([v.lower() for v in value_list]) - set(mapping.keys())
#     if unmapped:
#         print(f"Warning: The following values could not be mapped and will be assigned 0: {unmapped}")

#     return np.array([mapping.get(v.lower(), 0) for v in value_list], dtype=int)

def encode_labels(grouped_file, value_list, output_dir, label_type):
    """
    Generic encoding function for cell types, organs, and diseases.
    This version handles case-insensitive matching by converting all values to lowercase.
    Additionally, it appends unmapped values to a CSV file in the output directory.
    """
    grouped_data = pd.read_csv(grouped_file)
    mapping = {}

    for _, row in grouped_data.iterrows():
        label = row["cluster"]
        values = eval(row["group_values"])
        for v in values:
            mapping[v.lower()] = label  # Convert to lowercase for case-insensitive

    unmapped = set([v.lower() for v in value_list]) - set(mapping.keys())

    if unmapped:
        print(f"Warning: The following values could not be mapped and will be assigned 0: {unmapped}")
        
        # Define the file path
        unmapped_file = os.path.join(output_dir, f"unmapped_{label_type}.csv")

        # Load existing unmapped values if the file exists
        if os.path.exists(unmapped_file):
            existing_unmapped_df = pd.read_csv(unmapped_file)
            existing_unmapped_set = set(existing_unmapped_df["unmapped_value"].str.lower())
        else:
            existing_unmapped_set = set()

        # Merge new unmapped values with existing ones and save
        all_unmapped = existing_unmapped_set.union(unmapped)
        unmapped_df = pd.DataFrame({"unmapped_value": list(all_unmapped)})
        unmapped_df.to_csv(unmapped_file, index=False)
        
        print(f"Unmapped values saved to {unmapped_file}")

    return np.array([mapping.get(v.lower(), 0) for v in value_list], dtype=int)

def encode_disease_status(disease_name):
    """
    Encode disease_status based on the disease name.
    """
    healthy_terms = ["general", "healthy", "normal", "unknown", "unclassified"]
    return 0 if disease_name.lower() in healthy_terms else 1

def min_max_normalize(data):
    """
    Apply Min-Max normalization to the data.
    """
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    # Avoid division by zero
    denom = np.where(data_max - data_min == 0, 1, data_max - data_min)
    return (data - data_min) / denom

def process_h5ad_files(root_dir,dataset_name, mapping_table_file, grouped_cell_types_file,
                       grouped_organs_file, grouped_diseases_file,
                       cell_type_obs, organ_obs, disease_obs, processed_data_dir):
    """
    Processes h5ad files and generates partitioned X.npy and Y.npy (with four label columns).
    """
    os.makedirs(processed_data_dir, exist_ok=True)
    mapping_df = pd.read_csv(mapping_table_file, usecols=["index", "original_index"], dtype={"original_index": str})
    mapping_df["original_index"] = mapping_df["original_index"].astype(str).str.strip()
    mapping_dict = mapping_df.groupby("original_index")["index"].apply(list).to_dict()

    for tissue in os.listdir(root_dir):
        tissue_path = os.path.join(root_dir, tissue)
        if not os.path.isdir(tissue_path):
            continue

        tissue_name = tissue.replace(" ", "_").lower()

        for disease in os.listdir(tissue_path):
            disease_path = os.path.join(tissue_path, disease)
            if not os.path.isdir(disease_path):
                continue

            disease_name = disease.replace(" ", "_").lower()
            print(f"Processing: {tissue_name} / {disease_name}")

            tissue_output_dir = os.path.join(processed_data_dir, tissue_name)
            disease_output_dir = os.path.join(tissue_output_dir, disease_name)
            os.makedirs(disease_output_dir, exist_ok=True)

            partitions = get_partitioned_files(disease_path, PARTITION_SIZE_MB)
            disease_status = encode_disease_status(disease_name)

            for partition_idx, partition_files in enumerate(partitions):
                print(f"  Processing Partition {partition_idx} for {disease_name}...")

                disease_x, disease_y = [], []

                for h5ad_path in partition_files:
                    print(f"    Loading {os.path.basename(h5ad_path)}...")
                    adata = ad.read_h5ad(h5ad_path)
                    cell_type_list = adata.obs[cell_type_obs].tolist()
                    organ_list = adata.obs[organ_obs].tolist()
                    disease_list = adata.obs[disease_obs].tolist()
                    original_gene_ids = adata.var.index.astype(str).str.strip().tolist()

                    expression_matrix = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
                    num_cells, num_genes_expanded = expression_matrix.shape[0], len(mapping_df)
                    expanded_matrix = np.zeros((num_cells, num_genes_expanded), dtype=np.float16)

                    for original_idx, original_gene_id in enumerate(original_gene_ids):
                        if original_gene_id in mapping_dict:
                            for mapped_idx in mapping_dict[original_gene_id]:
                                expanded_matrix[:, mapped_idx] = expression_matrix[:, original_idx]

                    expanded_matrix = min_max_normalize(expanded_matrix)

                    y_cell_type = encode_labels(grouped_cell_types_file, cell_type_list, processed_data_dir, "cell_types")
                    y_organ = encode_labels(grouped_organs_file, organ_list, processed_data_dir, "organs")
                    y_disease = encode_labels(grouped_diseases_file, disease_list, processed_data_dir, "diseases")
                    y_disease_status = np.full((num_cells,), disease_status, dtype=int)

                    combined_y = np.vstack((y_cell_type, y_organ, y_disease, y_disease_status)).T

                    disease_x.append(expanded_matrix)
                    disease_y.append(combined_y)

                if disease_x:
                    final_x = np.concatenate(disease_x, axis=0)
                    print(f"    Final X shape: {final_x.shape}")
                    final_y = np.concatenate(disease_y, axis=0)
                    print(f"    Final Y shape: {final_y.shape}")

                    x_output_file = os.path.join(disease_output_dir, f"{dataset_name}_{disease_name}_X_partition_{partition_idx}.npy")
                    y_output_file = os.path.join(disease_output_dir, f"{dataset_name}_{disease_name}_Y_partition_{partition_idx}.npy")
                    np.save(x_output_file, final_x)
                    np.save(y_output_file, final_y)

                    print(f"  Saved {x_output_file} and {y_output_file}")


def process_dataset(dataset_name, biomed_transcript_file, biomed_protein_file, output_dir,
                    mapping_table_file, grouped_cell_types_file, grouped_organs_file,
                    grouped_diseases_file, cell_type_obs, organ_obs, disease_obs):
    """
    Generate gene mapping table and process h5ad files.

    - If the dataset is "cellxgene", it uses generate_mapping_table_from_csv().
    - Otherwise, it uses generate_mapping_table_from_h5ad().
    """
    root_dir = f"{dataset_name}_seacell_processed_data/meta_cells"
    processed_data_dir = os.path.join(output_dir, "processed_data")

    # Generate the mapping table based on the dataset type
    if dataset_name == "cellxgene":
        cellxgene_var_file = "cellxgene_var_names.csv"
        gene_mapping.generate_mapping_table_from_csv(cellxgene_var_file, biomed_transcript_file, biomed_protein_file, output_dir)
    else:
        gene_mapping.generate_mapping_table_from_h5ad(root_dir, biomed_transcript_file, biomed_protein_file, output_dir)

    process_h5ad_files(root_dir, dataset_name, mapping_table_file, grouped_cell_types_file,
                       grouped_organs_file, grouped_diseases_file, cell_type_obs, organ_obs, disease_obs, processed_data_dir)

def main():
    datasets = ["cellxgene", "geo", "brain_sc", "SEA-AD"]  # List of datasets to process

    # Define common file paths
    biomed_transcript_file = "biomedgraphica_transcript.csv"
    biomed_protein_file = "biomedgraphica_protein.csv"
    grouped_cell_types_file = "grouped_cell_types.csv"
    grouped_organs_file = "grouped_organs.csv"
    grouped_diseases_file = "grouped_diseases.csv"
    cell_type_obs = "cell_type"
    organ_obs = "organ"
    disease_obs = "disease"

    # Iterate through the datasets and process each
    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")
        output_dir = f"{dataset_name}_output"
        mapping_table_file = os.path.join(output_dir, "mapping_table.csv")

        process_dataset(dataset_name, biomed_transcript_file, biomed_protein_file, output_dir,
                        mapping_table_file, grouped_cell_types_file, grouped_organs_file,
                        grouped_diseases_file, cell_type_obs, organ_obs, disease_obs)

if __name__ == "__main__":
    main()