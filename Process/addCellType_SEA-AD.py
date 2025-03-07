import os
import scanpy as sc
import celltypist
from celltypist import models
import scipy.sparse
import pandas as pd
import numpy as np
from pathlib import Path

def should_process_file(input_file, output_dir):
    """Check if file needs processing"""
    input_name = Path(input_file).stem
    output_file = os.path.join(output_dir, f"{input_name}_labeled.h5ad")
    
    # Skip if output file exists and is not empty
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"Skipping {input_file} - already processed")
        return False
    return True

def process_h5_file(input_file, output_dir):
    """Process a single h5 file with CellTypist"""

    if not should_process_file(input_file, output_dir):
        return True

    print(f"Processing: {input_file}")
    
    try:
        # Read the 10x h5 file
        data = sc.read_10x_h5(input_file)

        data.raw = data.copy()

        sc.pp.filter_cells(data, min_genes=100)
        sc.pp.normalize_total(data, target_sum=1e4, inplace=True)
        sc.pp.log1p(data)
        
        # Add observation annotations
        data.obs['disease'] = "alzheimer's disease"
        data.obs['organ'] = 'brain'
        data.obs['substructure'] = 'unknown'
    
        data.var["Feature Name"] = data.var.index
        
        # Print some statistics
        print(f"Number of cells: {data.n_obs}")
        print(f"Number of genes: {data.n_vars}")
        # print(f"Number of matching genes with model: {len(common_genes)}")
        
        # Annotate using CellTypist
        predictions = celltypist.annotate(data,
                                        model="Adult_Human_PrefrontalCortex.pkl",
                                        majority_voting=True)
        labeled_data = predictions.to_adata()
        
        # Create output filename
        input_name = Path(input_file).stem
        output_file = os.path.join(output_dir, f"{input_name}_labeled.h5ad")
        
        # Save the labeled data
        labeled_data.write(output_file)
        print(f"Successfully saved: {output_file}\n")
        return True
        
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}\n")
        return False

def main():
    # Base directory
    base_dir = "results_SEA-AD"
    
    # Create output directory structure
    output_base = "labeled_results_SEA-AD"
    os.makedirs(output_base, exist_ok=True)
    
    # Keep track of processing results
    results = {
        'successful': [],
        'failed': []
    }
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        # Process h5 files
        h5_files = [f for f in files if 'raw_feature_bc_matrix.h5' in f]
        
        if h5_files:
            # Create corresponding output directory
            rel_path = os.path.relpath(root, base_dir)
            output_dir = os.path.join(output_base, rel_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Process each h5 file
            for h5_file in h5_files:
                input_path = os.path.join(root, h5_file)
                success = process_h5_file(input_path, output_dir)
                
                if success:
                    results['successful'].append(input_path)
                else:
                    results['failed'].append(input_path)
    
    # Print summary
    print("\nProcessing Summary:")
    print(f"Successfully processed: {len(results['successful'])} files")
    print(f"Failed to process: {len(results['failed'])} files")
    
    if results['failed']:
        print("\nFailed files:")
        for file in results['failed']:
            print(f"- {file}")
        
    # Save processing results to a log file
    log_file = os.path.join(output_base, "processing_log.txt")
    with open(log_file, 'w') as f:
        f.write("Processing Results\n")
        f.write("=================\n\n")
        f.write("Successful:\n")
        for file in results['successful']:
            f.write(f"+ {file}\n")
        f.write("\nFailed:\n")
        for file in results['failed']:
            f.write(f"- {file}\n")

if __name__ == "__main__":
    main()