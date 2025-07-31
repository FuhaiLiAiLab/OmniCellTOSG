import os
import numpy as np
import pandas as pd
import anndata as ad
from collections import defaultdict


def build_id_lookup(biomed_transcript_file: str) -> dict[str, list[str]]:
    df = pd.read_csv(
        biomed_transcript_file,
        usecols=["BioMedGraphica_Conn_ID", "HGNC_Symbol", "Ensembl_Gene_ID"],
        dtype=str
    )

    melted = (
        df.melt(id_vars="BioMedGraphica_Conn_ID",
                value_vars=["HGNC_Symbol", "Ensembl_Gene_ID"],
                value_name="index_id")
          .dropna(subset=["index_id"])
    )

    melted["index_id"] = melted["index_id"].str.strip().str.upper()

    lookup = defaultdict(list)
    for _, row in melted.iterrows():
        lookup[row["index_id"]].append(row["BioMedGraphica_Conn_ID"])

    return dict(lookup)


def generate_mapping_table_from_csv(cellxgene_var_file, biomed_transcript_file, biomed_protein_file, output_path):
    if os.path.exists(output_path):
        print(f"[Skip] {os.path.basename(output_path)} already exists, skipping generation.")
        return

    print(f"Generating mapping table at: {output_path}")

    # Load Biomed Transcript Data
    biomed_transcript_df = pd.read_csv(
        biomed_transcript_file,
        usecols=["BioMedGraphica_Conn_ID", "Ensembl_Gene_ID"],
        dtype=str
    )

    # Load CellxGene Variable Data
    cellxgene_df = pd.read_csv(
        cellxgene_var_file,
        usecols=["feature_id", "soma_joinid"],
        dtype=str
    )
    cellxgene_df["feature_id"] = cellxgene_df["feature_id"].str.strip()

    # Merge: keep all ensembl gene IDs and feature IDs mapping
    merged_df = biomed_transcript_df.merge(
        cellxgene_df,
        left_on="Ensembl_Gene_ID",
        right_on="feature_id",
        how="left"
    )

    # Build transcript DataFrame
    transcript_df = merged_df[["feature_id", "soma_joinid", "BioMedGraphica_Conn_ID"]].copy()
    transcript_df = transcript_df.fillna("")
    transcript_df["soma_joinid"] = transcript_df["soma_joinid"].replace("", pd.NA).astype("Int64")

    # Remove duplicates in case of multiple genes mapping to same BMGC_ID
    transcript_df = transcript_df.drop_duplicates("BioMedGraphica_Conn_ID", keep="first")

    # Sort for consistency
    transcript_df = transcript_df.sort_values(by="BioMedGraphica_Conn_ID").reset_index(drop=True)

    # Load Biomed Protein Data
    biomed_protein_df = pd.read_csv(
        biomed_protein_file,
        usecols=["BioMedGraphica_Conn_ID"],
        dtype=str
    ).drop_duplicates()
    biomed_protein_df["feature_id"] = ""
    biomed_protein_df["soma_joinid"] = pd.NA

    # Combine transcript + protein
    final_df = pd.concat([
        transcript_df[["feature_id", "soma_joinid", "BioMedGraphica_Conn_ID"]],
        biomed_protein_df[["feature_id", "soma_joinid", "BioMedGraphica_Conn_ID"]],
    ], ignore_index=True)

    # Add index column
    final_df.insert(0, "index", range(len(final_df)))
    final_df.columns = ["index", "original_id", "original_index", "BioMedGraphica_Conn_ID"]
    final_df["original_index"] = final_df["original_index"].astype("Int64")

    # Save to CSV
    final_df.to_csv(output_path, index=False)
    print(f"Mapping table saved successfully at: {output_path}")


def generate_mapping_table_from_h5ad_single(
    h5ad_path: str,
    id_lookup: dict[str, list[str]],  # 1 to many BMG id mapping
    biomed_transcript_file: str,
    biomed_protein_file: str,
    output_path: str,
):  
    
    expected_rows = 533_458

    # if os.path.exists(output_path):
    #     print(f"[Skip] {os.path.basename(output_path)} already exists, skipping generation.")
    #     return

    # Read genes from h5ad
    adata = ad.read_h5ad(h5ad_path)
    if adata.var.index.duplicated().any():
        raise ValueError(f"Found duplicated gene names in {h5ad_path}")
    gene_ids = pd.Series(adata.var.index.astype(str), name="original_id")

    def map_one(gid: str) -> list[str]:
        gid_clean = gid.strip().upper()
        if gid_clean.startswith("ENSG"):
            gid_clean = gid_clean.split(".")[0]
        return id_lookup.get(gid_clean, [])

    # Expand mapping into multiple rows
    records = []
    for original_id in gene_ids:
        for bmgc_id in map_one(original_id):
            records.append({
                "original_id": original_id,
                "original_index": original_id,
                "BioMedGraphica_Conn_ID": bmgc_id
            })

    matched_df = pd.DataFrame.from_records(records)
    matched_df = matched_df.drop_duplicates("BioMedGraphica_Conn_ID", keep="first")

    # Read all transcript BMGC IDs
    transcript_df = pd.read_csv(
        biomed_transcript_file,
        usecols=["BioMedGraphica_Conn_ID"],
        dtype=str
    ).drop_duplicates()

    # Merge all transcript IDs with matched info
    merged_transcript = transcript_df.merge(
        matched_df,
        on="BioMedGraphica_Conn_ID",
        how="left"
    )
    merged_transcript["original_id"] = merged_transcript["original_id"].fillna("")
    merged_transcript["original_index"] = merged_transcript["original_index"].fillna("")

    # Read protein IDs
    protein_df = pd.read_csv(
        biomed_protein_file,
        usecols=["BioMedGraphica_Conn_ID"],
        dtype=str
    ).drop_duplicates()
    protein_df["original_id"] = ""
    protein_df["original_index"] = ""

    # Combine transcript + protein
    final_df = pd.concat([
        merged_transcript[["original_id", "original_index", "BioMedGraphica_Conn_ID"]],
        protein_df[["original_id", "original_index", "BioMedGraphica_Conn_ID"]],
    ], ignore_index=True)

    # Add integer index
    final_df.insert(0, "index", range(len(final_df)))

    # Save
    final_df.to_csv(output_path, index=False)
    print(f"Mapping table written to {output_path}")

    if len(final_df) != expected_rows:
        raise ValueError(
            f"[{h5ad_path}] ❌ mapping table has {len(final_df)} rows, expected {expected_rows} rows. "
            f"Something went wrong with mapping."
        )


def construct_description_and_edge_index(biomed_transcript_file, biomed_protein_file, output_dir):

    # Define file paths
    biomed_transcript_description_file = biomed_transcript_file.replace(".csv", "_Description_Combined.csv")
    biomed_protein_description_file = biomed_protein_file.replace(".csv", "_Description_Combined.csv")
    relation_file = biomed_transcript_file.replace("_Transcript.csv", "_Relation.csv")

    x_descriptions_output_file = os.path.join(output_dir, "X_descriptions.npy")
    edge_index_output_file = os.path.join(output_dir, "edge_index.npy")
    ppi_edge_index_output_file = os.path.join(output_dir, "ppi_edge_index.npy")
    internal_edge_index_output_file = os.path.join(output_dir, "internal_edge_index.npy")

    mapping_file_path = os.path.join(output_dir,"mapping_table.csv")
    entity_mapping = pd.read_csv(mapping_file_path)

    entity_mapping.columns = ["index", "original_id", "original_index", "BioMedGraphica_Conn_ID"]

    # transcript_description = pd.read_csv(biomed_transcript_description_file)
    # protein_description = pd.read_csv(biomed_protein_description_file)

    # merged_data = entity_mapping.merge(transcript_description, on="BioMedGraphica_Conn_ID", how="left")
    # merged_data = merged_data.merge(protein_description, on="BioMedGraphica_Conn_ID", how="left", suffixes=("_transcript", "_protein"))
    # merged_data["Description"] =  merged_data["Description_transcript"].fillna("") + \
    #                             merged_data["Description_protein"].fillna("")

    # # Drop unnecessary columns
    # descriptions = merged_data[["Description"]]

    # descriptions.to_csv("s_desc.csv", index=False)

    # # Convert to numpy array with shape (n, 1)
    # descriptions_array = descriptions.to_numpy()

    # np.save(x_descriptions_output_file, descriptions_array)

    # print(f"Descriptions saved to {x_descriptions_output_file}")

    # Load edge data
    edge_data = pd.read_csv(relation_file)
    edge_data = edge_data[["BMGC_From_ID", "BMGC_To_ID", "Type"]]

    # Filter edge data based on entity_mapping
    filtered_edge_data = edge_data[
        edge_data["BMGC_From_ID"].isin(entity_mapping["BioMedGraphica_Conn_ID"]) &
        edge_data["BMGC_To_ID"].isin(entity_mapping["BioMedGraphica_Conn_ID"])
    ]

    # Map BioMedGraphica_Conn_ID to Index
    id_to_index = dict(zip(entity_mapping["BioMedGraphica_Conn_ID"], entity_mapping["index"]))

    filtered_edge_data = filtered_edge_data.copy()
    filtered_edge_data["From_Index"] = filtered_edge_data["BMGC_From_ID"].map(id_to_index)
    filtered_edge_data["To_Index"] = filtered_edge_data["BMGC_To_ID"].map(id_to_index)

    # Sort by From_Index and then by To_Index
    filtered_edge_data.sort_values(by=["From_Index", "To_Index"], inplace=True)

    # filtered_edge_data.to_csv("filtered_edge_data.csv", index=False)

    # Construct edge_index array
    edge_index = filtered_edge_data[["From_Index", "To_Index"]].dropna().astype(int).T.values

    # Save edge_index to npy file
    np.save(edge_index_output_file, edge_index)
    print(f"Edge index saved to {edge_index_output_file} with shape {edge_index.shape}")

    # Split into PPI and Internal
    ppi_edges = filtered_edge_data[filtered_edge_data["Type"] == "Protein-Protein"]
    internal_edges = filtered_edge_data[filtered_edge_data["Type"] != "Protein-Protein"]

    # Save PPI edge index
    ppi_edge_index = ppi_edges[["From_Index", "To_Index"]].T.values
    np.save(ppi_edge_index_output_file, ppi_edge_index)
    print(f"PPI edge index saved to {ppi_edge_index_output_file} with shape {ppi_edge_index.shape}")

    # Save Internal edge index
    internal_edge_index = internal_edges[["From_Index", "To_Index"]].T.values
    np.save(internal_edge_index_output_file, internal_edge_index)
    print(f"Internal edge index saved to {internal_edge_index_output_file} with shape {internal_edge_index.shape}")
