import os
import pandas as pd
from tqdm import tqdm

def load_mapping(mapping_path: str) -> pd.DataFrame:

    mdf = pd.read_csv(mapping_path)

    bmg_id_col = "BioMedGraphica_Conn_ID"
    name_col = "Name"

    mdf[name_col] = mdf[name_col].fillna("")
    mdf[name_col] = mdf[name_col].where(mdf[name_col].str.strip() != "", mdf[bmg_id_col])

    # Add index column
    mdf = mdf.reset_index(drop=True)
    mdf.insert(0, "index", range(len(mdf)))

    return mdf[["index", bmg_id_col, name_col]]

def _iter_sample_folders(results_dir: str):
    sample_folders = [
        f for f in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, f)) and f.startswith("sample_")
    ]
    sample_folders.sort()
    for folder in sample_folders:
        yield folder, os.path.join(results_dir, folder)


def _map_edge_names(edge_df: pd.DataFrame, idx2name: pd.Series) -> pd.DataFrame:

    for col in ["from", "to"]:
        edge_df[col] = pd.to_numeric(edge_df[col], errors="raise")

    edge_df["From_name"] = edge_df["from"].map(idx2name)
    edge_df["To_name"] = edge_df["to"].map(idx2name)

    return edge_df

def compute_edge_aggregated_sum(results_dir: str, mapping_path: str):

    mapping_df = load_mapping(mapping_path)
    idx2name = mapping_df.set_index("index")["Name"]

    out_dir = os.path.join(results_dir, "edge_aggregated_sum")
    os.makedirs(out_dir, exist_ok=True)

    for folder, folder_path in tqdm(list(_iter_sample_folders(results_dir)), desc="Edge SUM"):
        csv_path = os.path.join(folder_path, "last_layer.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df = _map_edge_names(df, idx2name)

        grouped = (
            df.groupby(["From_name", "To_name"], dropna=False, as_index=False)["value"]
              .sum()
              .rename(columns={"value": "value"})
        )

        out_file = os.path.join(out_dir, f"{folder}_edges.csv")
        grouped.to_csv(out_file, index=False)


def compute_edge_aggregated_avg(results_dir: str, mapping_path: str):

    mapping_df = load_mapping(mapping_path)
    idx2name = mapping_df.set_index("index")["Name"]

    out_dir = os.path.join(results_dir, "edge_aggregated_avg")
    os.makedirs(out_dir, exist_ok=True)

    for folder, folder_path in tqdm(list(_iter_sample_folders(results_dir)), desc="Edge AVG"):
        csv_path = os.path.join(folder_path, "last_layer.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df = _map_edge_names(df, idx2name)

        grouped = (
            df.groupby(["From_name", "To_name"], dropna=False, as_index=False)["value"]
              .mean()
              .rename(columns={"value": "value"})
        )

        out_file = os.path.join(out_dir, f"{folder}_edges.csv")
        grouped.to_csv(out_file, index=False)


def compute_node_attention_sum(edge_path: str, output_dir: str):

    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(edge_path) if f.endswith(".csv")]
    files.sort()

    for f in tqdm(files, desc="Node SUM"):
        df = pd.read_csv(os.path.join(edge_path, f))

        # compute in / out
        in_w = df.groupby("To_name", dropna=False)["value"].sum().reset_index()
        in_w.columns = ["node", "in_attention"]

        out_w = df.groupby("From_name", dropna=False)["value"].sum().reset_index()
        out_w.columns = ["node", "out_attention"]

        node_w = pd.merge(in_w, out_w, on="node", how="outer").fillna(0.0)
        node_w["avg_attention"] = (node_w["in_attention"] + node_w["out_attention"]) / 2.0

        out_file = os.path.join(output_dir, f.replace("_edges.csv", "_node_attentions.csv"))
        node_w.to_csv(out_file, index=False)


def compute_node_attention_avg(edge_path: str, output_dir: str):

    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(edge_path) if f.endswith(".csv")]
    files.sort()

    for f in tqdm(files, desc="Node AVG"):
        df = pd.read_csv(os.path.join(edge_path, f))

        in_w = df.groupby("To_name", dropna=False)["value"].mean().reset_index()
        in_w.columns = ["node", "in_attention"]

        out_w = df.groupby("From_name", dropna=False)["value"].mean().reset_index()
        out_w.columns = ["node", "out_attention"]

        node_w = pd.merge(in_w, out_w, on="node", how="outer").fillna(0.0)
        node_w["avg_attention"] = (node_w["in_attention"] + node_w["out_attention"]) / 2.0

        out_file = os.path.join(output_dir, f.replace("_edges.csv", "_node_attentions.csv"))
        node_w.to_csv(out_file, index=False)
 
def main():
    results_dir = "./CellTOSG_analysis_results/disease/Crohn_disease/gat/epoch_50_3_0.0005_2025_20250928_173207"
    mapping_path = "./mapping_table.csv"

    compute_edge_aggregated_sum(results_dir, mapping_path)
    # compute_edge_aggregated_avg(results_dir, mapping_path)

    edge_sum_dir = os.path.join(results_dir, "edge_aggregated_sum")
    edge_avg_dir = os.path.join(results_dir, "edge_aggregated_avg")

    node_sum_out = os.path.join(results_dir, "node_attentions_sum")
    node_avg_out = os.path.join(results_dir, "node_attentions_avg")

    # compute_node_attention_sum(edge_path=edge_sum_dir, output_dir=node_sum_out)
    compute_node_attention_avg(edge_path=edge_sum_dir, output_dir=node_avg_out)

    print("All done.")


if __name__ == "__main__":
    main()
