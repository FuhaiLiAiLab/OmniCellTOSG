import os
import scanpy as sc
import celltypist
from celltypist import models
import scipy.sparse
import pandas as pd
from pathlib import Path
import json

# Directory paths
input_dir = './Organ_Type'
output_dir = './Organ_Type_labeled'
checkpoint_dir = './checkpoints'  
os.makedirs(checkpoint_dir, exist_ok=True)  
checkpoint_file = os.path.join(checkpoint_dir, 'processed_files.json')

# Load GSM mapping CSV
csv_path = './final_standardized_geo_list_lowercase.csv' 
gsm_mapping = pd.read_csv(csv_path)

# Load previously processed files if checkpoint exists
try:
    with open(checkpoint_file, 'r') as f:
        processed_files = set(json.load(f))
    print(f"Loaded {len(processed_files)} previously processed files from checkpoint")
except FileNotFoundError:
    processed_files = set()
    print("Starting fresh processing - no checkpoint found")

# Create dictionaries for mapping GSM IDs to models and metadata
gsm_to_model = dict(zip(gsm_mapping['ID'], gsm_mapping['Model']))
gsm_to_disease = dict(zip(gsm_mapping['ID'], gsm_mapping['Disease_Type']))
gsm_to_organ = dict(zip(gsm_mapping['ID'], gsm_mapping['Organ_Type']))
gsm_to_substructure = dict(zip(gsm_mapping['ID'], gsm_mapping['substructure']))

# Walk through all directories and process .h5ad files
for root, dirs, files in os.walk(input_dir):

    rel_path = os.path.relpath(root, input_dir)
    current_output_dir = os.path.join(output_dir, rel_path)
    
    # Ensure output directory exists
    os.makedirs(current_output_dir, exist_ok=True)
    
    # Process each .h5ad file in the current directory
    for file in files:
        if file.endswith('_processed.h5ad'):
            gsm_id = file.split('_')[0]  # Extract GSM ID from the filename
            output_file = os.path.join(current_output_dir, f"{gsm_id}_labeled.h5ad")
            
            # Skip if file was already successfully processed
            if output_file in processed_files and os.path.exists(output_file):
                print(f"Skipping already processed file: {file}")
                continue
                
            model = gsm_to_model.get(gsm_id)  # Get the corresponding model
            
            if model:
                try:
                    # Read the .h5ad file
                    input_path = os.path.join(root, file)
                    data = sc.read_h5ad(input_path)
                    
                    # Preprocess the data
                    data.X = scipy.sparse.csr_matrix(data.X)
                    data.raw = data.copy()
                    
                    # Add observation annotations
                    data.obs['disease'] = gsm_to_disease.get(gsm_id)
                    data.obs['organ'] = gsm_to_organ.get(gsm_id)
                    data.obs['substructure'] = gsm_to_substructure.get(gsm_id)
                    
                    sc.pp.normalize_total(data, target_sum=1e4)
                    sc.pp.log1p(data)
                    data.var["Feature Name"] = data.var.index

                    # Annotate using CellTypist
                    predictions = celltypist.annotate(data, model=model, majority_voting=True)
                    labeled_data = predictions.to_adata()

                    # Save the labeled data
                    labeled_data.write(output_file)
                    print(f"Successfully processed and saved: {output_file}")
                    
                    # Add to processed files and save checkpoint
                    processed_files.add(output_file)
                    with open(checkpoint_file, 'w') as f:
                        json.dump(list(processed_files), f)

                except Exception as e:
                    print(f"Error processing {file}: {e}")
            else:
                print(f"No model found for {gsm_id}. Skipping file: {file}.")

print(f"Processing complete! Total files processed: {len(processed_files)}")