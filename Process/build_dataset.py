# build_dataset.py

import os
from pathlib import Path
import glob
import pandas as pd
import numpy as np
import anndata as ad
import gene_mapping


SOURCE_NAME_MAP = {
    "cellxgene": "CellxGene",
    "geo": "GEO",
    "heart_cell_atlas": "Heart Cell Atlas",
    "braincellatlas": "Brain Cell Atlas",
    "hca": "Human Cell Atlas",
    "hepatitisCatlas": "Hepatitis Atlas",
}

COLUMN_ORDER = [
    "source",
    "dataset_id",
    "suspension_type",
    "cell_type",
    "tissue_general",
    "tissue",
    "disease",
    "development_stage",
    "sex",
    "assay",
    "batch",
    "donor_id",
    "n_raw_cells",
    "raw_file",
    "raw_file_indices",
    "matrix_file_path",
    "matrix_row_idx",
]

N_COL = 412039
MAX_CELLS_PER_PART = 10_000


def _clean_obs_value(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s


def get_obs_value_str(adata, row_i: int, candidates: list[str], default: str = "unknown") -> str:
    for key in candidates:
        if key and key in adata.obs.columns:
            v = _clean_obs_value(adata.obs[key].iloc[row_i])
            if v is not None:
                return v
    return default


def _normalize_var_key(dataset_name: str, gid: str) -> str:
    if dataset_name == "cellxgene":
        s = str(gid).strip()
        if s == "" or s.lower() == "nan":
            return ""
        return s
    return gene_mapping.normalize_gene_symbol(gid)


def _load_mapping_dict(map_csv_path: str) -> dict[str, list[int]]:
    df = pd.read_csv(map_csv_path, dtype={"original_index": str}, usecols=["index", "original_index"])
    df["original_index"] = df["original_index"].astype("string").str.strip().str.upper()
    df["index"] = df["index"].astype(int)
    return df.groupby("original_index")["index"].apply(list).to_dict()


def process_multiple_datasets(
    datasets,
    bmg_feature_list_json_path,
    dataset_id_obs,
    suspension_type_obs,
    cell_type_obs,
    tissue_general_obs,
    tissue_obs,
    disease_obs,
    development_stage_obs,
    sex_obs,
    n_raw_cells_obs,
    raw_file_obs,
    raw_file_indices_obs,
    output_dir,
    overwrite_datasets=None,
):
    overwrite_datasets = set(overwrite_datasets or [])

    os.makedirs(output_dir, exist_ok=True)
    matrix_dir = os.path.join(output_dir, "expression_matrix")
    os.makedirs(matrix_dir, exist_ok=True)

    mapping_table_dir = os.path.join(output_dir, "mapping_tables")
    os.makedirs(mapping_table_dir, exist_ok=True)

    meta_file = os.path.join(output_dir, "cell_metadata.parquet")
    if os.path.exists(meta_file):
        old_meta = pd.read_parquet(meta_file)
        overwrite_sources = {SOURCE_NAME_MAP.get(ds, ds) for ds in overwrite_datasets}
        meta_kept = old_meta[~old_meta["source"].isin(overwrite_sources)].copy()
    else:
        meta_kept = pd.DataFrame(columns=["source"])

    if os.path.isdir(matrix_dir):
        for ds in overwrite_datasets:
            for path in glob.glob(os.path.join(matrix_dir, f"{ds}_*.npy")):
                os.remove(path)

    log_path = os.path.join(output_dir, "processing_log.csv")
    metadata_records = []
    part_buffers, part_ids, row_counters = {}, {}, {}

    with open(log_path, "w") as log_file:
        log_file.write("dataset,tissue_general,tissue,disease,file_path,n_cells\n")

        for dataset_name in datasets:
            dataset_mapping_table_dir = os.path.join(mapping_table_dir, dataset_name)
            os.makedirs(dataset_mapping_table_dir, exist_ok=True)

            src_name = SOURCE_NAME_MAP.get(dataset_name, dataset_name)
            if dataset_name not in overwrite_datasets and src_name in meta_kept["source"].unique():
                print(f"Skip {dataset_name} (already processed)")
                continue

            print(f"\nProcessing dataset: {dataset_name}")
            root_dir = f"../{dataset_name}_metacell_processed_data/meta_cells"

            global_mapping_dict = None
            if dataset_name == "cellxgene":
                var_file_path = "./cellxgene_var_id.csv"
                global_map_csv_path = os.path.join(dataset_mapping_table_dir, "mapping_table_cellxgene.csv")
                gene_mapping.generate_mapping_table_from_csv(
                    var_file_path,
                    bmg_feature_list_json_path,
                    global_map_csv_path,
                    n_col=N_COL,
                )
                global_mapping_dict = _load_mapping_dict(global_map_csv_path)

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

                            if dataset_name == "cellxgene":
                                mapping_dict = global_mapping_dict
                            else:
                                map_csv_path = os.path.join(
                                    dataset_mapping_table_dir,
                                    f"mapping_table_{dataset_name}_"
                                    f"{tissue_general}_{tissue}_{disease}_"
                                    f"{Path(file).stem}.csv",
                                )
                                gene_mapping.generate_mapping_table_from_h5ad_single(
                                    h5ad_path,
                                    bmg_feature_list_json_path,
                                    map_csv_path,
                                    n_col=N_COL,
                                )
                                mapping_dict = _load_mapping_dict(map_csv_path)

                            print(f"Processing {h5ad_path}...")
                            adata = ad.read_h5ad(h5ad_path)
                            n_cells = adata.shape[0]
                            log_file.write(
                                f"{dataset_name},{tissue_general},{tissue},{disease},{h5ad_path},{n_cells}\n"
                            )
                            log_file.flush()

                            raw_var = adata.var.index.astype(str).tolist()
                            var_keys = [_normalize_var_key(dataset_name, g) for g in raw_var]

                            X = adata.X
                            if isinstance(X, np.ndarray):
                                X_dense = X
                                get_col = lambda j: X_dense[:, j]
                            else:
                                try:
                                    X_csc = X.tocsc()
                                    get_col = lambda j: np.asarray(X_csc[:, j].toarray()).ravel()
                                except Exception:
                                    X_dense = X.toarray()
                                    get_col = lambda j: X_dense[:, j]

                            expanded = np.zeros((n_cells, N_COL), dtype=np.float32)

                            for col, key_name in enumerate(var_keys):
                                if not key_name:
                                    continue
                                idxs = mapping_dict.get(key_name.upper())
                                if not idxs:
                                    continue
                                col_vec = get_col(col).astype(np.float32, copy=False)
                                for idx in idxs:
                                    expanded[:, int(idx)] = col_vec

                            buffer = part_buffers[key]
                            for i in range(expanded.shape[0]):
                                buffer.append(expanded[i])

                                part_id = part_ids[key]
                                row_idx = row_counters[key]
                                npy_name = f"{dataset_name}_{tissue_general}_part_{part_id}.npy"

                                record = {
                                    "source": src_name,
                                    "tissue_general": tissue_general,
                                    "matrix_file_path": npy_name,
                                    "matrix_row_idx": row_idx,
                                }

                                record["dataset_id"] = get_obs_value_str(adata, i, [dataset_id_obs], default="unknown")
                                record["suspension_type"] = get_obs_value_str(adata, i, [suspension_type_obs], default="unknown")
                                record["cell_type"] = get_obs_value_str(adata, i, [cell_type_obs], default="unknown")
                                record["development_stage"] = get_obs_value_str(adata, i, [development_stage_obs], default="unknown")
                                record["sex"] = get_obs_value_str(adata, i, [sex_obs], default="unknown")

                                # record["tissue_general"] = get_obs_value_str(adata, i, [tissue_general_obs], default=str(tissue_general))
                                record["tissue"] = get_obs_value_str(adata, i, [tissue_obs], default=str(tissue))
                                record["disease"] = get_obs_value_str(adata, i, [disease_obs], default=str(disease))

                                record["n_raw_cells"] = get_obs_value_str(adata, i, [n_raw_cells_obs], default="unknown")
                                record["raw_file"] = get_obs_value_str(adata, i, [raw_file_obs], default="unknown")
                                record["raw_file_indices"] = get_obs_value_str(adata, i, [raw_file_indices_obs], default="unknown")

                                record["assay"] = get_obs_value_str(adata, i, ["assay", "Assay"], default="unknown")
                                record["batch"] = get_obs_value_str(adata, i, ["batch", "Batch", "SampleBatch", "processBatch", "Seqbatch"], default="unknown")
                                record["donor_id"] = get_obs_value_str(adata, i, ["donor_id", "donor", "Donor", "individual", "participant_id"], default="unknown")

                                metadata_records.append(record)

                                row_counters[key] += 1

                                if len(buffer) >= MAX_CELLS_PER_PART:
                                    np.save(os.path.join(matrix_dir, npy_name), np.vstack(buffer))
                                    print(f"Saved {npy_name} with {len(buffer)} cells")
                                    buffer.clear()
                                    row_counters[key] = 0
                                    part_ids[key] += 1

                buffer = part_buffers[key]
                if buffer:
                    part_id = part_ids[key]
                    npy_name = f"{dataset_name}_{tissue_general}_part_{part_id}.npy"
                    np.save(os.path.join(matrix_dir, npy_name), np.vstack(buffer))
                    print(f"Final saved {npy_name} with {len(buffer)} cells (end of {tissue_general})")
                    buffer.clear()

    ordered_meta = pd.DataFrame(metadata_records)
    all_meta = pd.concat([meta_kept, ordered_meta], ignore_index=True)

    for c in COLUMN_ORDER:
        if c not in all_meta.columns:
            all_meta[c] = "unknown"

    extra_cols = [c for c in all_meta.columns if c not in COLUMN_ORDER]
    all_meta = all_meta[COLUMN_ORDER + extra_cols]

    all_meta["matrix_row_idx"] = all_meta["matrix_row_idx"].astype(int)
    all_meta.to_parquet(os.path.join(output_dir, "cell_metadata.parquet"), index=False)
    all_meta.to_csv(os.path.join(output_dir, "cell_metadata.csv"), index=False)
    print("Saved unified metadata parquet and CSV")


def main():
    datasets = ["cellxgene", "braincellatlas", "geo", "hca", "hepatitisCatlas"]
    overwrite = []

    bmg_feature_list_json_path = "./bmg_feature_list.json"

    dataset_id_obs = "dataset_id"
    suspension_type_obs = "suspension_type"
    cell_type_obs = "cell_type"
    tissue_general_obs = "tissue_general"
    tissue_obs = "tissue"
    disease_obs = "disease"
    development_stage_obs = "development_stage"
    sex_obs = "sex"
    n_raw_cells_obs = "n_raw_cells"
    raw_file_obs = "raw_file"
    raw_file_indices_obs = "raw_file_indices"

    output_dir = "../CellTOSG_dataset_v3"
    # output_dir = "../../../CellTOSG_dataset_v2"

    process_multiple_datasets(
        datasets,
        bmg_feature_list_json_path,
        dataset_id_obs,
        suspension_type_obs,
        cell_type_obs,
        tissue_general_obs,
        tissue_obs,
        disease_obs,
        development_stage_obs,
        sex_obs,
        n_raw_cells_obs,
        raw_file_obs,
        raw_file_indices_obs,
        output_dir,
        overwrite_datasets=overwrite,
    )


if __name__ == "__main__":
    main()
