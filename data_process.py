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

def encode_cell_types(grouped_clusters_file, cell_type_list):
    """
    Encodes cell types into cluster labels based on the provided grouping file.
    
    Parameters:
    - grouped_clusters_file: Path to the CSV file containing cell type to cluster mappings.
    - cell_type_list: List of cell types to encode.

    Returns:
    - A NumPy array of encoded cluster labels.
    """
    grouped_clusters = pd.read_csv(grouped_clusters_file)

    # Create cell_type -> cluster mapping
    cluster_mapping = {}
    for _, row in grouped_clusters.iterrows():
        cluster = row["cluster"]
        cell_types = eval(row["Cell Type"])  # Convert stored string list into an actual list
        for cell_type in cell_types:
            cluster_mapping[cell_type] = cluster

    return np.array([cluster_mapping.get(ct, -1) for ct in cell_type_list], dtype=int)

def process_h5ad_files(root_dir, mapping_table_file, grouped_clusters_file, cell_type_obs, processed_data_dir):
    """
    Processes all h5ad files within tissue-disease structured directories, partitioning them into groups of X MB,
    and generates X.npy and Y.npy for each partition, storing them in a dedicated disease folder.
    
    Parameters:
    - root_dir: Root directory containing tissue-disease structured folders.
    - mapping_table_file: Path to the mapping table CSV file.
    - grouped_clusters_file: Path to the grouped clusters CSV file.
    - processed_data_dir: Directory to store processed X.npy and Y.npy files.
    """
    os.makedirs(processed_data_dir, exist_ok=True)
    mapping_df = pd.read_csv(mapping_table_file, usecols=["index", "original_index"], dtype={"original_index": str})
    mapping_df["original_index"] = mapping_df["original_index"].astype(str).str.strip()
    mapping_dict = mapping_df.groupby("original_index")["index"].apply(list).to_dict()

    for tissue in os.listdir(root_dir):
        tissue_path = os.path.join(root_dir, tissue)
        if not os.path.isdir(tissue_path):
            continue

        tissue_name = tissue.replace(" ", "_")

        for disease in os.listdir(tissue_path):
            disease_path = os.path.join(tissue_path, disease)
            if not os.path.isdir(disease_path):
                continue

            disease_name = disease.replace(" ", "_")
            print(f"Processing: {tissue_name} / {disease_name}")

            # Create output directories
            tissue_output_dir = os.path.join(processed_data_dir, tissue_name)
            disease_output_dir = os.path.join(tissue_output_dir, disease_name)
            os.makedirs(disease_output_dir, exist_ok=True)

            # Get partitioned files
            partitions = get_partitioned_files(disease_path, PARTITION_SIZE_MB)

            for partition_idx, partition_files in enumerate(partitions):
                print(f"  Processing Partition {partition_idx} for {disease_name}...")

                disease_x, disease_y = [], []

                for h5ad_path in partition_files:
                    print(f"    Loading {os.path.basename(h5ad_path)}...")
                    adata = ad.read_h5ad(h5ad_path)
                    cell_type_list = adata.obs[cell_type_obs].tolist()
                    # cell_type_list = adata.obs["majority_voting"].tolist()
                    original_gene_ids = adata.var.index.astype(str).str.strip().tolist()

                    expression_matrix = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
                    num_cells, num_genes_expanded = expression_matrix.shape[0], len(mapping_df)
                    expanded_matrix = np.zeros((num_cells, num_genes_expanded), dtype=np.float16)

                    for original_idx, original_gene_id in enumerate(original_gene_ids):
                        if original_gene_id in mapping_dict:
                            for mapped_idx in mapping_dict[original_gene_id]:
                                expanded_matrix[:, mapped_idx] = expression_matrix[:, original_idx]

                    y_labels = encode_cell_types(grouped_clusters_file, cell_type_list)

                    disease_x.append(expanded_matrix)
                    disease_y.append(y_labels)

                if disease_x:
                    final_x, final_y = np.concatenate(disease_x, axis=0), np.concatenate(disease_y, axis=0)

                    # Generate partitioned X and Y filenames
                    x_output_file = os.path.join(disease_output_dir, f"{disease_name}_X_partition_{partition_idx}.npy")
                    y_output_file = os.path.join(disease_output_dir, f"{disease_name}_Y_partition_{partition_idx}.npy")

                    np.save(x_output_file, final_x)
                    np.save(y_output_file, final_y)

                    print(f"  Saved {x_output_file} and {y_output_file}")

def main():
    dataset_name = "cellxgene"
    root_dir = f"{dataset_name}_meta_cells"
    output_dir = f"{dataset_name}_output"
    processed_data_dir = os.path.join(output_dir, "processed_data")

    biomed_transcript_file = "biomedgraphica_transcript.csv"
    biomed_protein_file = "biomedgraphica_protein.csv"

    mapping_table_file = os.path.join(output_dir, "mapping_table.csv")
    grouped_clusters_file = "grouped_clusters.csv"
    cell_type_obs = "cell_type"

    ## dataset_name | cell_type_obs

    # cellxgene "cell_type"
    cellxgene_var_file = "cellxgene_var_names.csv"
    gene_mapping.generate_mapping_table_from_csv(cellxgene_var_file, biomed_transcript_file, biomed_protein_file, output_dir)

    # # geo "majority_voting"
    # # SEA-AD "majority_voting"
    # # brain_sc "cell_type"
    # gene_mapping.generate_mapping_table_from_h5ad(root_dir, biomed_transcript_file, biomed_protein_file, output_dir)

    process_h5ad_files(root_dir, mapping_table_file, grouped_clusters_file, cell_type_obs, processed_data_dir)

if __name__ == "__main__":
    main()
