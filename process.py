import os
import numpy as np
import pandas as pd
import scanpy as sc

import os
import numpy as np
import pandas as pd
import torch

from torch_geometric.data import Data, Batch
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

def add_meta_cell_id(dir_path: str = "./sc_meta_cell/cell/pan-cancer/", 
                     var_dir_path: str = "./sc_meta_cell/cell/",
                     output_dir_path: str = "./sc_meta_cell/output/output_ENSG_id"):
    """
    Map ID to SEACell h5ad files

    Args:
        dir_path: str
            The path to the directory containing the SEACell files.
        output_dir_path: str
            The path to the output directory.
    """
    os.makedirs(output_dir_path, exist_ok=True)

    var_name_data = pd.read_csv(var_dir_path + "cellxgene_var_names.csv")
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".h5ad"):
            print(f"Processing {file_name}")
            adata = sc.read_h5ad(os.path.join(dir_path, file_name))
            sparse_matrix = adata.layers["raw"]
            dense_matrix = sparse_matrix.toarray()
            # sea-cell sample ID
            sample_ids = adata.obs.index 
            df = pd.DataFrame(dense_matrix)
            df.insert(0, "sample_id", sample_ids)
            df.columns = ["sample_id"] + var_name_data["feature_id"].tolist()
            output_path = os.path.join(output_dir_path, file_name.replace("_SEAcells.h5ad", "_ENSG_id.csv"))
            df.to_csv(output_path, index=False)
            print(f"Saved {output_path}")


def count_cell_types(dir_path: str = "./sc_meta_cell/metrics/pan-cancer/",
                     output_dir_path: str = "./sc_meta_cell/output/"):
    """
    Count cell types across all files in the given directory.
    Args:
        dir_path: Path to the directory containing the files.
        output_dir_path: Path to the output directory.
    """
    # Lists to store results
    all_cell_types = []
    unique_cell_types = set()
    filewise_unique_cell_types = {}

    # Iterate over files in the folder
    for filename in os.listdir(dir_path):
        # Check if the file is a CSV file and ends with "_purity.csv"
        if filename.endswith("_purity.csv"):
            file_path = os.path.join(dir_path, filename)
            # Read the file
            try:
                df = pd.read_csv(file_path)
                # Check if the "cell_type" column exists
                if "cell_type" in df.columns:
                    cell_types = df["cell_type"].tolist()
                    all_cell_types.extend(cell_types)
                    unique_cell_types.update(cell_types)
                    # Get unique values for each file
                    filewise_unique_cell_types[filename] = set(cell_types)
                else:
                    print(f"File {filename} does not contain a \"cell_types\" column.")
            except Exception as e:
                print(f"Error reading file {filename}: {e}")

    # Save all results to files
    all_cell_types_file = os.path.join(output_dir_path, "all_cell_types.csv")
    unique_cell_types_file = os.path.join(output_dir_path, "unique_cell_types.csv")
    filewise_unique_cell_types_file = os.path.join(output_dir_path, "filewise_unique_cell_types.csv")

    # Save all cell types
    pd.DataFrame(all_cell_types, columns=["cell_type"]).to_csv(all_cell_types_file, index=False)
    # Save unique cell types
    pd.DataFrame(list(unique_cell_types), columns=["cell_type"]).to_csv(unique_cell_types_file, index=False)
    # Save unique cell types for each file
    filewise_data = []
    for filename, unique_cells in filewise_unique_cell_types.items():
        for cell in unique_cells:
            filewise_data.append({"filename": filename, "cell_type": cell})
    filewise_df = pd.DataFrame(filewise_data)
    filewise_df.to_csv(filewise_unique_cell_types_file, index=False)

    # Print results with counts
    print("All cell types across all files:")
    print(f"Total count: {len(all_cell_types)}")
    print(all_cell_types)

    print("Unique cell types across all files:")
    print(f"Total unique count: {len(unique_cell_types)}")
    print(list(unique_cell_types))

    print("Unique cell types for each file:")
    for filename, unique_cells in filewise_unique_cell_types.items():
        print(f"File: {filename}")
        print(f"Unique count: {len(unique_cells)}")
        print(unique_cells)


###################### Merge Cell Types ########################
def merge_cell_types(output_dir_path: str = "./sc_meta_cell/output/"):
    """
    Merge cell types based on a mapping dictionary.
    Args:
        adata: AnnData object containing cell type annotations.
        cell_type_mapping: Dictionary mapping original cell types to new cell types.
    """

    # Load the dataset
    unique_cell_df = pd.read_csv(output_dir_path + "unique_cell_types.csv")
    # Preprocess: lowercase and strip whitespace
    unique_cell_df["cell_type_clean"] = unique_cell_df["cell_type"].str.lower().str.strip()
    # Convert cell types to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(unique_cell_df["cell_type_clean"])

    # clustering with Euclidean distance
    clustering_model = AgglomerativeClustering(
        n_clusters=None,  # Let the distance threshold determine clusters
        linkage="average",  # Use average linkage for clustering
        distance_threshold=1  # Adjust this value to control cluster granularity
    )
    labels = clustering_model.fit_predict(tfidf_matrix.toarray())

    # Add the cluster labels to the dataframe
    unique_cell_df["cluster"] = labels
    # Group cell types by clusters
    grouped_clusters = unique_cell_df.groupby("cluster")["cell_type"].apply(list).reset_index()
    # Save the grouped clusters to a CSV file
    grouped_clusters.to_csv(output_dir_path + "grouped_clusters.csv", index=False)
    # Print the number of clusters
    print(f"Total clusters: {len(grouped_clusters)}")


def map_cell_types(output_dir_path: str = "./sc_meta_cell/output"):
    '''
    Map clustered cell types to cell type labels

    Args:
        output_dir_path: Path to the output directory
    '''
    # Define folders and files
    pan_cancer_metric_folder = "./sc_meta_cell/metrics/pan-cancer/"
    y_data_folder = output_dir_path + "/Y_data"
    grouped_clusters_file = output_dir_path + "/grouped_clusters.csv"

    os.makedirs(y_data_folder, exist_ok=True)

    # Load grouped clusters
    print(f"Loading grouped_clusters_file: {grouped_clusters_file}")
    grouped_clusters = pd.read_csv(grouped_clusters_file)

    # Create a mapping from cell_type to cluster
    cluster_mapping = {}
    for _, row in grouped_clusters.iterrows():
        cluster = row["cluster"]
        cell_types = eval(row["cell_type"])  # Convert string to list
        for cell_type in cell_types:
            cluster_mapping[cell_type] = cluster
    print(f"Loaded cluster mapping: {cluster_mapping}")

    # Process files in the pan-cancer_metric folder
    for filename in os.listdir(pan_cancer_metric_folder):
        if filename.endswith("_purity.csv"):
            file_path = os.path.join(pan_cancer_metric_folder, filename)
            print(f"Processing file: {filename}")

            try:
                # Read the CSV file
                data = pd.read_csv(file_path)

                # Sort by the first column, extracting numerical part if applicable
                sorted_data = data.sort_values(
                    by=data.columns[0],
                    key=lambda x: x.str.extract(r"SEACell-(\d+)", expand=False).astype(float) 
                    if x.str.contains("SEACell-").any() else x
                )

                # Map cell_type to cluster using the loaded cluster_mapping
                if "cell_type" in sorted_data.columns:
                    sorted_data["cluster"] = sorted_data["cell_type"].map(cluster_mapping)
                else:
                    print(f"Warning: \"cell_type\" column not found in {filename}. Skipping cluster mapping.")

                # Save to Y_data folder
                output_file_path = os.path.join(y_data_folder, filename)
                sorted_data.to_csv(output_file_path, index=False)
                print(f"Saved sorted and encoded file to {output_file_path}")

            except Exception as e:
                print(f"Error processing file {filename}: {e}")


def sort_label(output_dir_path: str = "./sc_meta_cell/output"):
    '''
    Sort the Y/label data files in the output folder by the first column

    Args:
        output_dir_path (str): The path to the output folder
    '''
    # Define folders and files
    pan_cancer_metric_folder = "./sc_meta_cell/metrics/pan-cancer/"
    y_data_folder = output_dir_path + "/Y_data"
    grouped_clusters_file = output_dir_path + "/grouped_clusters.csv"

    os.makedirs(y_data_folder, exist_ok=True)

    # Load grouped clusters
    print(f"Loading grouped_clusters_file: {grouped_clusters_file}")
    grouped_clusters = pd.read_csv(grouped_clusters_file)

    # Create a mapping from cell_type to cluster
    cluster_mapping = {}
    for _, row in grouped_clusters.iterrows():
        cluster = row["cluster"]
        cell_types = eval(row["cell_type"])  # Convert string to list
        for cell_type in cell_types:
            cluster_mapping[cell_type] = cluster
    print(f"Loaded cluster mapping: {cluster_mapping}")

    # Process files in the pan-cancer_metric folder
    for filename in os.listdir(pan_cancer_metric_folder):
        if filename.endswith("_purity.csv"):
            file_path = os.path.join(pan_cancer_metric_folder, filename)
            print(f"Processing file: {filename}")

            try:
                # Read the CSV file
                data = pd.read_csv(file_path)

                # Sort by the first column, extracting numerical part if applicable
                sorted_data = data.sort_values(
                    by=data.columns[0],
                    key=lambda x: x.str.extract(r"SEACell-(\d+)", expand=False).astype(float) 
                    if x.str.contains("SEACell-").any() else x
                )

                # Map cell_type to cluster using the loaded cluster_mapping
                if "cell_type" in sorted_data.columns:
                    sorted_data["cluster"] = sorted_data["cell_type"].map(cluster_mapping)
                else:
                    print(f"Warning: \"cell_type\" column not found in {filename}. Skipping cluster mapping.")

                # Save to Y_data folder
                output_file_path = os.path.join(y_data_folder, filename)
                sorted_data.to_csv(output_file_path, index=False)
                print(f"Saved sorted and encoded file to {output_file_path}")

            except Exception as e:
                print(f"Error processing file {filename}: {e}")


def edge_construction(root_dir_path: str = "./sc_meta_cell/output",
                      database_dir_path: str = "./BioMedGraphica/"):
    # Define the input and output paths
    input_folder = root_dir_path + "/output_ENSG_id"
    mapping_output_folder = root_dir_path + "/mapping_tables"
    output_folder = root_dir_path + "/X_data"
    # Ensure the output folders exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(mapping_output_folder, exist_ok=True)

    # Define and read the entity files
    transcript_entity_file = database_dir_path + "Entity/Transcript/biomedgraphica_transcript.csv"
    transcript_entity_data = pd.read_csv(transcript_entity_file)
    protein_entity_file = database_dir_path + "/Entity/Protein/biomedgraphica_protein.csv"
    protein_entity_data = pd.read_csv(protein_entity_file)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            print(f"Processing file: {filename}")

            # import pdb; pdb.set_trace()
            try:
                # Read the CSV file
                data = pd.read_csv(file_path)
                # Transpose the data
                transposed_data = data.set_index(data.columns[0]).T.reset_index()
                transposed_data.rename(columns={"index": "Ensembl_Gene_ID"}, inplace=True)

                # Merge with transcript_entity_data
                mapping_columns = ["Ensembl_Gene_ID", "BioMedGraphica_ID"]
                transcript_mapping = transcript_entity_data[mapping_columns]
                merged_data = pd.merge(
                    transposed_data,
                    transcript_mapping,
                    on="Ensembl_Gene_ID",
                    how="inner"
                )

                # Sort by BioMedGraphica_ID
                merged_data.sort_values(by="BioMedGraphica_ID", inplace=True)

                # Add BioMedGraphica_ID from protein_entity_data
                protein_ids = protein_entity_data[["BioMedGraphica_ID"]].copy()
                protein_ids.loc[:, transposed_data.columns[1:]] = 0  # Fill protein sample values with 0

                # Append the protein IDs to the sorted transcript data
                final_combined_data = pd.concat([merged_data, protein_ids],ignore_index=True,sort=False)

                # Extract the mapping table
                mapping_table = final_combined_data[["Ensembl_Gene_ID", "BioMedGraphica_ID"]].drop_duplicates()
                mapping_table.rename(columns={"Ensembl_Gene_ID": "Original_ID"}, inplace=True)
                mapping_table.insert(0, "Index", range(len(mapping_table)))  # Add Index as the first column

                # Save the mapping table
                mapping_table_path = os.path.join(mapping_output_folder, f"{filename.replace('.csv', '_mapping_table.csv')}")
                mapping_table.to_csv(mapping_table_path, index=False)

                # Drop the "Ensembl_Gene_ID" column after merging
                final_combined_data.drop(columns=["Ensembl_Gene_ID"], inplace=True, errors="ignore")

                # Ensure "BioMedGraphica_ID" is the first column
                reordered_columns = ["BioMedGraphica_ID"] + [col for col in final_combined_data.columns if col != "BioMedGraphica_ID"]
                reordered_data = final_combined_data[reordered_columns]

                # Transpose the data back
                final_data = reordered_data.set_index("BioMedGraphica_ID").T.reset_index()
                final_data.rename(columns={"index": "Sample_ID"}, inplace=True)

                # Sort the final transposed data by the first column (Sample_ID)
                # final_data.sort_values(by="Sample_ID", key=lambda x: x.str.extract(r'SEACell-(/d+)', expand=False).astype(int),inplace=True)
                final_data.sort_values(by="Sample_ID",key=lambda x: x.str.extract(r'SEACell-(\d+)', expand=False).fillna(-1).astype(int),inplace=True)

                # Save the final processed file
                output_file_path = os.path.join(output_folder, f"{filename.replace('.csv', '_processed.csv')}")
                final_data.to_csv(output_file_path, index=False)

                print(f"Processed file saved to {output_file_path}")
                print(f"Mapping table saved to {mapping_table_path}")

            except Exception as e:
                print(f"Error processing file {filename}: {e}")


def node_construction(root_dir_path: str = "./sc_meta_cell/output",
                      database_dir_path: str = "./BioMedGraphica"):
    # Define the input and output paths
    mapping_output_folder = root_dir_path + "/mapping_tables"
    os.makedirs(mapping_output_folder, exist_ok=True)

    # Define and read the entity files
    transcript_description_file = database_dir_path + "Entity/Transcript/biomedgraphica_transcript_description.csv"
    protein_description_file = database_dir_path + "/Entity/Protein/biomedgraphica_protein_description.csv"
    # Load description files
    protein_description = pd.read_csv(protein_description_file)
    transcript_description = pd.read_csv(transcript_description_file)

    edge_data_file = database_dir_path + "/Relation/biomedgraphica_relation.csv"

    output_x_descriptions_file = root_dir_path + "/X_descriptions.npy"
    edge_index_output_file = root_dir_path + "/edge_index.npy"

    # Read the first mapping table file
    mapping_files = sorted(os.listdir(mapping_output_folder))
    if not mapping_files:
        raise FileNotFoundError(f"No files found in {mapping_output_folder}.")
    mapping_file_path = os.path.join(mapping_output_folder, mapping_files[0])
    entity_mapping = pd.read_csv(mapping_file_path)

    # Ensure proper column names
    entity_mapping.columns = ["Index", "Original_ID", "BioMedGraphica_ID"]

    # Merge with protein description
    merged_data = entity_mapping.merge(protein_description, on="BioMedGraphica_ID", how="left")

    # Merge with transcript description (retain existing descriptions if present)
    merged_data = merged_data.merge(transcript_description, on="BioMedGraphica_ID", how="left", suffixes=("_protein", "_transcript"))

    # Combine descriptions from protein and transcript
    merged_data["Description"] = merged_data["Description_protein"].fillna("") + \
                                merged_data["Description_transcript"].fillna("")

    # Drop unnecessary columns
    descriptions = merged_data[["Description"]]

    # Convert to numpy array with shape (n, 1)
    descriptions_array = descriptions.to_numpy()

    # Save to .npy file
    os.makedirs(os.path.dirname(output_x_descriptions_file), exist_ok=True)
    np.save(output_x_descriptions_file, descriptions_array)

    print(f"Descriptions saved to {output_x_descriptions_file}")


    # Load edge data
    edge_data_raw = pd.read_csv(edge_data_file)
    edge_data = edge_data_raw[["From_ID", "To_ID", "Type"]].copy()

    # Filter edge data based on entity_mapping
    filtered_edge_data = edge_data[
        edge_data["From_ID"].isin(entity_mapping["BioMedGraphica_ID"]) &
        edge_data["To_ID"].isin(entity_mapping["BioMedGraphica_ID"])
    ]

    # Map BioMedGraphica_ID to Index
    id_to_index = dict(zip(entity_mapping["BioMedGraphica_ID"], entity_mapping["Index"]))
    filtered_edge_data["From_Index"] = filtered_edge_data["From_ID"].map(id_to_index)
    filtered_edge_data["To_Index"] = filtered_edge_data["To_ID"].map(id_to_index)

    # Sort by From_Index and then by To_Index
    filtered_edge_data.sort_values(by=["From_Index", "To_Index"], inplace=True)

    # Construct edge_index array
    edge_index = filtered_edge_data[["From_Index", "To_Index"]].dropna().astype(int).T.values

    # Save edge_index to npy file
    np.save(edge_index_output_file, edge_index)
    print(f"Edge index saved to {edge_index_output_file} with shape {edge_index.shape}")


def npy_gen(root_dir_path: str = "./sc_meta_cell/output"):
    '''
    Generate numpy array from csv files in the root directory

    Args:
        root_dir_path (str): Root directory path containing csv files
    '''

    # Define folders
    x_data_folder = root_dir_path + '/X_data'
    y_data_folder = root_dir_path + '/Y_data'
    edge_index_file = root_dir_path + '/edge_index.npy'  # Path to pre-generated edge_index file

    # # Load pre-generated edge_index (shared by all graphs)
    # edge_index = torch.tensor(np.load(edge_index_file), dtype=torch.long)
    # print(f"Loaded edge_index with shape: {edge_index.shape}")

    # File paths
    x_file_path = os.path.join(x_data_folder, 'acute myeloid leukemia_bone marrow_partition_0_ENSG_id_processed.csv')
    y_file_path = os.path.join(y_data_folder, 'acute myeloid leukemia_bone marrow_partition_0_purity.csv')

    import pdb; pdb.set_trace()

    # Load expression data (X) and label data (Y)
    x_data = pd.read_csv(x_file_path)  # Shape: [num_samples, num_genes]
    y_data = pd.read_csv(y_file_path)  # Shape: [num_samples, cluster/label]

    # Match samples by Sample_ID
    merged_data = pd.merge(
        x_data,
        y_data[['SEACell', 'cluster']],  # Use the cluster column for labels
        left_on='Sample_ID',
        right_on='SEACell',
        how='inner'
    )

    import pdb; pdb.set_trace()

    # Extract gene expression matrix and cluster labels
    expression_matrix = merged_data.drop(columns=['Sample_ID', 'SEACell', 'cluster']).values  # Shape: [num_samples, num_genes]
    cluster_labels = merged_data['cluster'].values  # Shape: [num_samples]

    # Create a graph for each sample in the file
    for sample_idx in range(expression_matrix.shape[0]):
        # Sample features: [num_genes]
        x = torch.tensor(expression_matrix[sample_idx], dtype=torch.float)

        # Graph label: [1] (use cluster as the label)
        y = torch.tensor([cluster_labels[sample_idx]], dtype=torch.long)




if __name__ == "__main__":
    # count_cell_types(dir_path = "./sc_meta_cell/metrics/pan-cancer/", output_dir_path = "sc_meta_cell/output/")
    # merge_cell_types()
    # map_cell_types()
    # sort_label()
    # edge_construction()
    # node_construction()
    npy_gen()