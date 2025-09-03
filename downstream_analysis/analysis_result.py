import os
import pandas as pd
from tqdm import tqdm

results_dir = "./CellTOSG_analysis_results/disease/Alzheimers_Disease/gat/epoch_50_3_0.0005_2025_20250812_122041"

output_dir = os.path.join(results_dir, "node_weights_results")
os.makedirs(output_dir, exist_ok=True)

sample_folders = [f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f)) and f.startswith("sample_")]

for folder in tqdm(sample_folders, desc="Processing samples"):
    sample_path = os.path.join(results_dir, folder)
    if os.path.isdir(sample_path) and folder.startswith("sample_"):
        csv_path = os.path.join(sample_path, "last_layer.csv")
        if os.path.exists(csv_path):
            print(f"Processing {csv_path} ...")
            
            df = pd.read_csv(csv_path)
            
            # compute in and out weights
            in_weight = df.groupby("to")["value"].sum().reset_index()
            in_weight.columns = ["node", "in_weight"]
            
            out_weight = df.groupby("from")["value"].sum().reset_index()
            out_weight.columns = ["node", "out_weight"]
            
            node_weights = pd.merge(in_weight, out_weight, on="node", how="outer").fillna(0)
            
            out_file = os.path.join(output_dir, f"{folder}_node_weights.csv")
            node_weights.to_csv(out_file, index=False)
