import os
import pandas as pd
import scanpy as sc
from scipy import sparse
from scRNA_workflow import *


data_dir = "./GSM_datasets" 

gene_list_path = "./OS_scRNA_gene_index.19264.tsv"  

results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

gene_list_df = pd.read_csv(gene_list_path, delimiter="\t")
gene_list = list(gene_list_df["gene_name"])

# Define processing functions
def process_10x_data(data_path, output_path):
    try:
        # Read 10x data
        adata = sc.read_10x_mtx(data_path)
        
        # Uniform gene names
        X_df = pd.DataFrame(
            sparse.csr_matrix.toarray(adata.X), 
            index=adata.obs.index.tolist(), 
            columns=adata.var.index.tolist()
        )
        X_df, to_fill_columns, var = main_gene_selection(X_df, gene_list)
        adata_uni = sc.AnnData(X_df)
        adata_uni.obs = adata.obs
        adata_uni.uns = adata.uns
        
        # Quality control (without drawing)
        adata_uni = BasicFilter(adata_uni, qc_min_genes=100, qc_min_cells=0, plot_show=False)
        
        # Skip saving if the dataset has 0 cells
        if adata_uni.shape[0] == 0:
            # print(f"Warning: Dataset at {data_path} has 0 cells after filtering. Skipping save.")
            return


        save_adata_h5ad(adata_uni, output_path)

    except Exception as e:
        print(f"Error processing 10x data at {data_path}: {e}")

def process_csv_data(csv_path, output_path):
    try:
 
        adata = pd.read_csv(csv_path, index_col=0)
        
        # Uniform gene names
        X_df, to_fill_columns, var = main_gene_selection(adata.T, gene_list)
        adata_uni = sc.AnnData(X_df)
        
        # Quality control (without drawing)
        adata_uni = BasicFilter(adata_uni, qc_min_genes=100, qc_min_cells=0, plot_show=False)
        
        # Skip saving if the dataset has 0 cells
        if adata_uni.shape[0] == 0:
            print(f"Warning: Dataset at {csv_path} has 0 cells after filtering. Skipping save.")
            return

        save_adata_h5ad(adata_uni, output_path)

    except Exception as e:
        print(f"Error processing CSV data at {csv_path}: {e}")

# Disable all plotting in BasicFilter and QC_Metrics_info
def BasicFilter(adata, qc_min_genes=100, qc_min_cells=0, plot_show=False):
    """
    Perform basic filtering without plotting.
    """
    # print('Before filter, %d Cells, %d Genes' % (adata.shape))
    sc.pp.filter_cells(adata, min_genes=qc_min_genes)
    sc.pp.filter_genes(adata, min_cells=qc_min_cells)
    # print('After filter, %d Cells, %d Genes' % (adata.shape))
    return adata

def QC_Metrics_info(adata, doublet_removal=False, plot_show=False):
    """
    Perform QC metrics calculation without plotting.
    """
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # Annotate mitochondrial genes
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    return adata

# Loop through all datasets in the directory
for dataset in os.listdir(data_dir):
    dataset_path = os.path.join(data_dir, dataset)
    
    # Skip files or invalid directories
    if not os.path.isdir(dataset_path):
        print(f"Skipping invalid entry: {dataset}")
        continue
    
    if "matrix.mtx.gz" in os.listdir(dataset_path):
        # Process 10x data
        output_path = os.path.join(results_dir, f"{dataset}_processed.h5ad")
        process_10x_data(dataset_path, output_path)
    elif any(file.endswith(".csv.gz") for file in os.listdir(dataset_path)):
        # Find the CSV file
        csv_file = next(file for file in os.listdir(dataset_path) if file.endswith(".csv.gz"))
        csv_path = os.path.join(dataset_path, csv_file)
        # Process CSV data
        output_path = os.path.join(results_dir, f"{dataset}_processed.h5ad")
        process_csv_data(csv_path, output_path)
