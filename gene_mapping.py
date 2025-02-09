import os
import numpy as np
import pandas as pd
import anndata as ad

def get_one_h5ad_file(root_dir):
    """
    Searches for one .h5ad file in the root_dir's tissue/disease structure and returns its path.

    Parameters:
    - root_dir: The root directory containing tissue/disease subfolders.

    Returns:
    - Path to the first found .h5ad file, or None if no file is found.
    """
    for tissue in os.listdir(root_dir):  # Iterate over tissue folders
        tissue_path = os.path.join(root_dir, tissue)
        if not os.path.isdir(tissue_path):
            continue  # Skip if not a directory

        for disease in os.listdir(tissue_path):  # Iterate over disease folders
            disease_path = os.path.join(tissue_path, disease)
            if not os.path.isdir(disease_path):
                continue  # Skip if not a directory

            # Find one .h5ad file
            for file in os.listdir(disease_path):
                if file.endswith(".h5ad"):
                    return os.path.join(disease_path, file)  # Return the first found h5ad file

    return None  # Return None if no h5ad file is found

def generate_mapping_table_from_csv(cellxgene_var_file, biomed_transcript_file, biomed_protein_file, output_dir, output_file="mapping_table.csv"):
    """
    Generates the mapping table if it does not already exist.
    
    Parameters:
    - cellxgene_var_file: Path to CellxGene variable information CSV file.
    - biomed_transcript_file: Path to Biomed Transcript CSV file.
    - biomed_protein_file: Path to Biomed Protein CSV file.
    - output_dir: Target directory to store the mapping table.
    - output_file: Output filename for the mapping table (default: "output_mapping_table.csv").
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    if os.path.exists(output_path):
        print(f"Mapping table already exists: {output_path}, skipping generation.")
        return

    print(f"Generating mapping table at: {output_path}")

    # Load Biomed Transcript Data
    biomed_transcript_df = pd.read_csv(biomed_transcript_file, usecols=["BioMedGraphica_ID", "Ensembl_Gene_ID"])

    # Load CellxGene Variable Data
    cellxgene_df = pd.read_csv(cellxgene_var_file, usecols=["feature_id", "soma_joinid"])

    # Merge on Ensembl_Gene_ID
    merged_df = biomed_transcript_df.merge(cellxgene_df, left_on="Ensembl_Gene_ID", right_on="feature_id", how="left")

    # Prepare final transcript dataframe
    transcript_final_df = merged_df[["feature_id", "soma_joinid", "BioMedGraphica_ID"]].copy()
    transcript_final_df = transcript_final_df.fillna("")
    transcript_final_df["soma_joinid"] = transcript_final_df["soma_joinid"].replace("", pd.NA).astype("Int64")
    transcript_final_df = transcript_final_df.sort_values(by="BioMedGraphica_ID").reset_index(drop=True)

    # Load Biomed Protein Data
    biomed_protein_df = pd.read_csv(biomed_protein_file, usecols=["BioMedGraphica_ID"])
    biomed_protein_df["feature_id"] = ""
    biomed_protein_df["soma_joinid"] = ""

    # Merge transcript and protein data
    final_df = pd.concat([transcript_final_df, biomed_protein_df[["feature_id", "soma_joinid", "BioMedGraphica_ID"]]], ignore_index=True)

    # Add index column
    final_df.insert(0, "index", range(len(final_df)))
    final_df.columns = ["index", "original_id", "original_index", "BioMedGraphica_ID"]
    final_df["original_index"] = final_df["original_index"].replace("", pd.NA).astype("Int64")

    # Save to CSV
    final_df.to_csv(output_path, index=False)
    construct_description_and_edge_index(biomed_transcript_file, biomed_protein_file, output_dir)
    print(f"Mapping table saved successfully at: {output_path}")

def generate_mapping_table_from_h5ad(root_dir, biomed_transcript_file, biomed_protein_file, output_dir, output_file="mapping_table.csv"):
    """
    Generates a mapping table from an h5ad file by mapping its index (HGNC_Symbol) 
    to BioMedGraphica's HGNC_Symbol, ensuring all transcript and protein mappings are included.

    Parameters:
    - root_dir: The root directory containing tissue/disease subfolders.
    - biomed_transcript_file: Path to the BioMedGraphica transcript CSV file containing HGNC_Symbol mappings.
    - biomed_protein_file: Path to the BioMedGraphica protein CSV file containing BioMedGraphica_ID.
    - output_dir: Target directory to store the mapping table.
    - output_file: Output filename for the mapping table (default: "mapping_table.csv").
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    # # Check if the mapping table already exists
    # if os.path.exists(output_path):
    #     print(f"Mapping table already exists: {output_path}, skipping generation.")
    #     return

    # Find an h5ad file from the tissue-disease structure
    h5ad_file = get_one_h5ad_file(root_dir)
    if h5ad_file is None:
        print("No h5ad file found in the given root directory.")
        return

    print(f"Generating mapping table from {h5ad_file} to {output_path}...")

    # Load h5ad file
    adata = ad.read_h5ad(h5ad_file)

    # Extract gene metadata from index (HGNC_Symbol is the index itself)
    gene_metadata = pd.DataFrame({"original_id": adata.var.index.to_list()})
    gene_metadata.to_csv("var_names.csv", index=False)
    gene_metadata["original_index"] = gene_metadata["original_id"]  # Same as original_id

    # Load Biomed Transcript Data (Ensure all transcript mappings are included)
    biomed_transcript_df = pd.read_csv(biomed_transcript_file, usecols=["BioMedGraphica_ID", "HGNC_Symbol"])

    # Merge transcript mapping, ensuring all BioMedGraphica_ID are included
    merged_df = biomed_transcript_df.merge(gene_metadata, left_on="HGNC_Symbol", right_on="original_index", how="left")

    # Ensure original_index is retained where available; otherwise, leave empty
    merged_df["original_id"].fillna("", inplace=True)
    merged_df["original_index"].fillna("", inplace=True)

    # Compute mapping statistics before adding protein data
    total_transcript_genes = len(merged_df)
    mapped_genes = merged_df["original_index"].str.strip().replace("", pd.NA).dropna().shape[0]  # Count non-empty original_index

    print(f"Mapped genes: {mapped_genes} / {total_transcript_genes}")

    # Keep only relevant columns
    transcript_final_df = merged_df[["original_id", "original_index", "BioMedGraphica_ID"]]

    # Load Biomed Protein Data (Ensure all BioMedGraphica_ID from proteins are included)
    biomed_protein_df = pd.read_csv(biomed_protein_file, usecols=["BioMedGraphica_ID"])
    biomed_protein_df["original_id"] = ""  # No mapping for proteins
    biomed_protein_df["original_index"] = ""

    # Merge transcript and protein data, ensuring all BioMedGraphica_ID are present
    final_df = pd.concat([transcript_final_df, biomed_protein_df[["original_id", "original_index", "BioMedGraphica_ID"]]], ignore_index=True)

    # Add index column
    final_df.insert(0, "index", range(len(final_df)))

    # Save to CSV
    final_df.to_csv(output_path, index=False)

    construct_description_and_edge_index(biomed_transcript_file, biomed_protein_file, output_dir)
    print(f"Mapping table saved successfully at: {output_path}")

def construct_description_and_edge_index(biomed_transcript_file, biomed_protein_file, output_dir):
    # Define file paths
    biomed_transcript_description_file = biomed_transcript_file.replace(".csv", "_description.csv")
    biomed_protein_description_file = biomed_protein_file.replace(".csv", "_description.csv")
    relation_file = biomed_transcript_file.replace("_transcript.csv", "_relation.csv")

    x_descriptions_output_file = os.path.join(output_dir, "X_descriptions.npy")
    edge_index_output_file = os.path.join(output_dir, "edge_index.npy")

    mapping_file_path = os.path.join(output_dir,"mapping_table.csv")
    entity_mapping = pd.read_csv(mapping_file_path)

    entity_mapping.columns = ["index", "original_id", "original_index", "BioMedGraphica_ID"]

    transcript_description = pd.read_csv(biomed_transcript_description_file)
    protein_description = pd.read_csv(biomed_protein_description_file)

    merged_data = entity_mapping.merge(transcript_description, on="BioMedGraphica_ID", how="left")
    merged_data = merged_data.merge(protein_description, on="BioMedGraphica_ID", how="left", suffixes=("_transcript", "_protein"))
    merged_data["Description"] =  merged_data["Description_transcript"].fillna("") + \
                                merged_data["Description_protein"].fillna("")

    # Drop unnecessary columns
    descriptions = merged_data[["Description"]]

    # descriptions.to_csv("descriptions.csv", index=False)

    # Convert to numpy array with shape (n, 1)
    descriptions_array = descriptions.to_numpy()

    np.save(x_descriptions_output_file, descriptions_array)

    print(f"Descriptions saved to {x_descriptions_output_file}")

    # Load edge data
    edge_data = pd.read_csv(relation_file)
    edge_data = edge_data[["From_ID", "To_ID", "Type"]]

    # Filter edge data based on entity_mapping
    filtered_edge_data = edge_data[
        edge_data["From_ID"].isin(entity_mapping["BioMedGraphica_ID"]) &
        edge_data["To_ID"].isin(entity_mapping["BioMedGraphica_ID"])
    ]

    # Map BioMedGraphica_ID to Index
    id_to_index = dict(zip(entity_mapping["BioMedGraphica_ID"], entity_mapping["index"]))

    filtered_edge_data = filtered_edge_data.copy()
    filtered_edge_data["From_Index"] = filtered_edge_data["From_ID"].map(id_to_index)
    filtered_edge_data["To_Index"] = filtered_edge_data["To_ID"].map(id_to_index)

    # Sort by From_Index and then by To_Index
    filtered_edge_data.sort_values(by=["From_Index", "To_Index"], inplace=True)

    # filtered_edge_data.to_csv("filtered_edge_data.csv", index=False)

    # Construct edge_index array
    edge_index = filtered_edge_data[["From_Index", "To_Index"]].dropna().astype(int).T.values

    # Save edge_index to npy file
    np.save(edge_index_output_file, edge_index)
    print(f"Edge index saved to {edge_index_output_file} with shape {edge_index.shape}")
