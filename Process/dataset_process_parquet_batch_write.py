import os
from pathlib import Path
import glob
import pandas as pd
import numpy as np
import anndata as ad
import gene_mapping

# readable names used in metadata
SOURCE_NAME_MAP = {
    "cellxgene": "CellxGene",
    "geo": "GEO",
    "heart_cell_atlas": "Heart Cell Atlas",
    "braincellatlas": "Brain Cell Atlas",
    "hepatitisatlas": "Hepatitis Atlas",

}

def process_multiple_datasets(
    datasets,
    biomed_transcript_file,
    biomed_protein_file,
    dataset_id_obs,
    suspension_type_obs,
    cell_type_obs,
    tissue_general_obs,
    tissue_obs,
    disease_obs,
    development_stage_obs,
    sex_obs,
    output_dir,
    overwrite_datasets=None,       # names in *datasets* that must be recomputed
):
    overwrite_datasets = set(overwrite_datasets or [])

    os.makedirs(output_dir, exist_ok=True)
    matrix_dir = os.path.join(output_dir, "expression_matrix")
    os.makedirs(matrix_dir, exist_ok=True)

    mapping_table_dir = os.path.join(output_dir, "mapping_tables")
    os.makedirs(mapping_table_dir, exist_ok=True)

    # 1) keep old metadata rows that are not overwritten
    meta_file = os.path.join(output_dir, "cell_metadata.parquet")
    if os.path.exists(meta_file):
        old_meta = pd.read_parquet(meta_file)
        overwrite_sources = {
            SOURCE_NAME_MAP.get(ds, ds) for ds in overwrite_datasets
        }
        meta_kept = old_meta[~old_meta["source"].isin(overwrite_sources)].copy()
    else:
        meta_kept = pd.DataFrame()

    # 2) delete npy blocks belonging to overwritten datasets
    if os.path.isdir(matrix_dir):
        for ds in overwrite_datasets:
            for path in glob.glob(os.path.join(matrix_dir, f"{ds}_*.npy")):
                os.remove(path)

    # 3) common variables
    log_path = os.path.join(output_dir, "processing_log.csv")
    metadata_records = []
    MAX_CELLS_PER_PART = 10_000
    part_buffers, part_ids, row_counters = {}, {}, {}

    # build transcript lookup only once
    id_lookup = gene_mapping.build_id_lookup(biomed_transcript_file)

    with open(log_path, "w") as log_file:
        log_file.write("dataset,tissue_general,tissue,disease,file_path,n_cells\n")

        for dataset_name in datasets:
            # skip datasets that are already in meta_kept and are *not* overwritten
            src_name = SOURCE_NAME_MAP.get(dataset_name, dataset_name)
            if (
                dataset_name not in overwrite_datasets
                and src_name in meta_kept["source"].unique()
            ):
                print(f"Skip {dataset_name} (already processed)")
                continue

            print(f"\nProcessing dataset: {dataset_name}")
            root_dir = (
                f"./processed_h5ad_data/{dataset_name}_metacell_processed_data/meta_cells"
            )

            # cellxgene → one mapping for the whole dataset
            if dataset_name == "cellxgene":
                var_file = "cellxgene_var_id.csv"
                global_map_csv = os.path.join(mapping_table_dir, "mapping_table_cellxgene.csv")
                gene_mapping.generate_mapping_table_from_csv(
                    var_file,
                    biomed_transcript_file,
                    biomed_protein_file,
                    global_map_csv
                )
                gdf = pd.read_csv(
                    global_map_csv,
                    usecols=["index", "original_index"],
                    dtype={"original_index": str},
                )
                gdf["original_index"] = gdf["original_index"].str.strip()
                global_mapping_dict = (
                    gdf.groupby("original_index")["index"].apply(list).to_dict()
                )
                global_num_genes = len(gdf)

            # traverse directory tree
            for tissue_general in os.listdir(root_dir):
                tg_path = os.path.join(root_dir, tissue_general)
                if not os.path.isdir(tg_path):
                    continue

                key = (tissue_general, dataset_name)
                part_buffers.setdefault(key, [])
                part_ids.setdefault(key, 0)
                row_counters.setdefault(key, 0)

                for tissue in os.listdir(tg_path):
                    tissue_path = os.path.join(tg_path, tissue)
                    if not os.path.isdir(tissue_path):
                        continue

                    for disease in os.listdir(tissue_path):
                        disease_path = os.path.join(tissue_path, disease)
                        if not os.path.isdir(disease_path):
                            continue

                        for file in os.listdir(disease_path):
                            if not file.endswith(".h5ad"):
                                continue

                            h5ad_path = os.path.join(disease_path, file)

                            # mapping for this file
                            if dataset_name == "cellxgene":
                                mapping_dict = global_mapping_dict
                                num_genes = global_num_genes
                            else:
                                map_csv = os.path.join(
                                    mapping_table_dir,
                                    f"mapping_table_{dataset_name}_"
                                    f"{tissue_general}_{tissue}_{disease}_"
                                    f"{Path(file).stem}.csv",
                                )
                                gene_mapping.generate_mapping_table_from_h5ad_single(
                                    h5ad_path,
                                    id_lookup,
                                    biomed_transcript_file,
                                    biomed_protein_file,
                                    map_csv,
                                )
                                mdf = pd.read_csv(
                                    map_csv,
                                    usecols=["index", "original_index"],
                                    dtype={"original_index": str},
                                )
                                mdf["original_index"] = mdf["original_index"].str.strip()
                                mapping_dict = (
                                    mdf.groupby("original_index")["index"]
                                    .apply(list)
                                    .to_dict()
                                )
                                num_genes = len(mdf)

                            # read and expand matrix
                            print(f"Reading {h5ad_path}...")
                            adata = ad.read_h5ad(h5ad_path)
                            n_cells = adata.shape[0]
                            log_file.write(
                                f"{dataset_name},{tissue_general},{tissue},"
                                f"{disease},{h5ad_path},{n_cells}\n"
                            )
                            log_file.flush()

                            gene_list = adata.var.index.astype(str).str.strip().tolist()
                            X = (
                                adata.X.toarray()
                                if not isinstance(adata.X, np.ndarray)
                                else adata.X
                            )
                            expanded = np.zeros((X.shape[0], num_genes), dtype=np.float32)
                            for col, gid in enumerate(gene_list):
                                if gid in mapping_dict:
                                    for idx in mapping_dict[gid]:
                                        expanded[:, idx] = X[:, col]

                            # buffer rows
                            buffer = part_buffers[key]
                            for i in range(expanded.shape[0]):
                                buffer.append(expanded[i])

                                part_id = part_ids[key]
                                row_idx = row_counters[key]
                                npy_name = f"{dataset_name}_{tissue_general}_part_{part_id}.npy"

                                metadata_records.append(
                                    {
                                        "source": src_name,
                                        "dataset_id": str(
                                            adata.obs[dataset_id_obs].iloc[i]
                                        ),
                                        "suspension_type": str(
                                            adata.obs[suspension_type_obs].iloc[i]
                                        ),
                                        "cell_type": str(
                                            adata.obs[cell_type_obs].iloc[i]
                                        ),
                                        "tissue_general": tissue_general,
                                        "tissue": str(adata.obs[tissue_obs].iloc[i]),
                                        "disease": str(adata.obs[disease_obs].iloc[i]),
                                        "development_stage": str(
                                            adata.obs[development_stage_obs].iloc[i]
                                        ),
                                        "sex": str(adata.obs[sex_obs].iloc[i]),
                                        "matrix_file_path": npy_name,
                                        "matrix_row_idx": row_idx,
                                    }
                                )

                                row_counters[key] += 1

                                if len(buffer) >= MAX_CELLS_PER_PART:
                                    np.save(
                                        os.path.join(matrix_dir, npy_name),
                                        np.vstack(buffer),
                                    )
                                    print(
                                        f"Saved {npy_name} with {len(buffer)} cells"
                                    )
                                    buffer.clear()
                                    row_counters[key] = 0
                                    part_ids[key] += 1

                # flush buffer for this tissue_general at the end
                buffer = part_buffers[key]
                if buffer:
                    part_id = part_ids[key]
                    npy_name = f"{dataset_name}_{tissue_general}_part_{part_id}.npy"
                    np.save(os.path.join(matrix_dir, npy_name), np.vstack(buffer))
                    print(
                        f"Final saved {npy_name} with {len(buffer)} cells "
                        f"(end of {tissue_general})"
                    )
                    buffer.clear()

    # 4) merge new metadata with kept part and write
    all_meta = pd.concat([meta_kept, pd.DataFrame(metadata_records)], ignore_index=True)
    all_meta.to_parquet(os.path.join(output_dir, "cell_metadata.parquet"), index=False)
    print("Saved unified metadata parquet")


def main():
    # datasets to process this run
    datasets = ["cellxgene", "braincellatlas", "geo", "hepatitisatlas"]

    # datasets that will be overwrited
    overwrite = ["cellxgene", "braincellatlas", "geo", "hepatitisatlas"]  # or set(["cellxgene", "braincellatlas"])

    biomed_transcript_file = "./BMG_Conn/BioMedGraphica_Conn_Transcript.csv"
    biomed_protein_file = "./BMG_Conn/BioMedGraphica_Conn_Protein.csv"

    dataset_id_obs = "dataset_id"
    suspension_type_obs = "suspension_type"
    cell_type_obs = "cell_type"
    tissue_general_obs = "tissue_general"
    tissue_obs = "tissue"
    disease_obs = "disease"
    development_stage_obs = "development_stage"
    sex_obs = "sex"

    output_dir = "./dataset_outputs"

    process_multiple_datasets(
        datasets,
        biomed_transcript_file,
        biomed_protein_file,
        dataset_id_obs,
        suspension_type_obs,
        cell_type_obs,
        tissue_general_obs,
        tissue_obs,
        disease_obs,
        development_stage_obs,
        sex_obs,
        output_dir,
        overwrite_datasets=overwrite,
    )


if __name__ == "__main__":
    main()
