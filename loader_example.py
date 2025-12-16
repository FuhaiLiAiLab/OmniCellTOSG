from CellTOSG_Loader_new import CellTOSGDataLoader
import numpy as np
import pandas as pd
import os

data_root = "./OmniCellTOSG_Dataset"
output_dir = "./test_results"
os.makedirs(output_dir, exist_ok=True)

# Test configurations
DISEASES = [
    "Alzheimer disease",
    "leukemia", 
    "pancreatic ductal adenocarcinoma",
    "lung adenocarcinoma"
]

CELL_TYPES = [
    "CD4-positive, alpha-beta T cell",
    "acinar cell",
    "pancreatic ductal cell"
]

# Combination tests: (disease, cell_type)
COMBINATIONS = [
    ("pancreatic ductal adenocarcinoma", "acinar cell"),
    ("pancreatic ductal adenocarcinoma", "pancreatic ductal cell"),
]

TOP_K = 100  # Number of top genes to return

def test_retrieval(conditions, task, label_column, test_name):
    """Test data retrieval with given conditions."""
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"Conditions: {conditions}")
    print(f"{'='*60}")
    
    try:
        dataset = CellTOSGDataLoader(
            root=data_root,
            conditions=conditions,
            task=task,
            label_column=label_column,
            sample_ratio=1.0,
            sample_size=None,
            shuffle=False,
            stratified_balancing=False,
            extract_mode="inference",
            random_state=2025,
            train_text=False,
            train_bio=False,
            dataset_correction=None,
            output_dir=os.path.join(output_dir, test_name.replace(" ", "_"))
        )
        
        X, Y, metadata = dataset.data, dataset.labels, dataset.metadata
        
        result = {
            "test_name": test_name,
            "conditions": str(conditions),
            "num_samples": X.shape[0] if X is not None else 0,
            "num_features": X.shape[1] if X is not None and len(X.shape) > 1 else 0,
            "success": True,
            "error": None
        }
        
        print(f"✓ Retrieved {result['num_samples']} samples with {result['num_features']} features")
        
        # Get top K genes by mean expression
        if X is not None and X.shape[0] > 0:
            mean_expr = np.mean(X, axis=0)
            top_k_indices = np.argsort(mean_expr)[-TOP_K:][::-1]
            top_k_values = mean_expr[top_k_indices]
            
            result["top_k_gene_indices"] = top_k_indices.tolist()
            result["top_k_gene_values"] = top_k_values.tolist()
            
            print(f"✓ Top {TOP_K} genes by mean expression computed")
            
        return result, X, Y, metadata
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return {
            "test_name": test_name,
            "conditions": str(conditions),
            "num_samples": 0,
            "num_features": 0,
            "success": False,
            "error": str(e)
        }, None, None, None


def main():
    all_results = []
    
    # Test 1: Disease retrieval
    print("\n" + "="*80)
    print("PART 1: DISEASE RETRIEVAL TESTS")
    print("="*80)
    
    for disease in DISEASES:
        result, X, Y, metadata = test_retrieval(
            conditions={"disease": disease},
            task="disease",
            label_column="disease",
            test_name=f"disease_{disease}"
        )
        all_results.append(result)
    
    # Test 2: Cell type retrieval
    print("\n" + "="*80)
    print("PART 2: CELL TYPE RETRIEVAL TESTS")
    print("="*80)
    
    for cell_type in CELL_TYPES:
        result, X, Y, metadata = test_retrieval(
            conditions={"cell_type": cell_type},
            task="cell_type",
            label_column="cell_type",
            test_name=f"celltype_{cell_type}"
        )
        all_results.append(result)
    
    # Test 3: Combination retrieval (disease + cell_type)
    print("\n" + "="*80)
    print("PART 3: COMBINATION RETRIEVAL TESTS (Disease + Cell Type)")
    print("="*80)
    
    for disease, cell_type in COMBINATIONS:
        result, X, Y, metadata = test_retrieval(
            conditions={"disease": disease, "cell_type": cell_type},
            task="disease",
            label_column="disease",
            test_name=f"combo_{disease}_{cell_type}"
        )
        all_results.append(result)
    
    # Save summary results
    print("\n" + "="*80)
    print("SUMMARY OF ALL TESTS")
    print("="*80)
    
    summary_df = pd.DataFrame([{
        "test_name": r["test_name"],
        "conditions": r["conditions"],
        "num_samples": r["num_samples"],
        "num_features": r["num_features"],
        "success": r["success"],
        "error": r["error"]
    } for r in all_results])
    
    summary_path = os.path.join(output_dir, "retrieval_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    
    # Save top K genes for successful tests
    top_genes_data = []
    for r in all_results:
        if r["success"] and "top_k_gene_indices" in r:
            for i, (idx, val) in enumerate(zip(r["top_k_gene_indices"], r["top_k_gene_values"])):
                top_genes_data.append({
                    "test_name": r["test_name"],
                    "rank": i + 1,
                    "gene_index": idx,
                    "mean_expression": val
                })
    
    if top_genes_data:
        top_genes_df = pd.DataFrame(top_genes_data)
        top_genes_path = os.path.join(output_dir, "top_k_genes.csv")
        top_genes_df.to_csv(top_genes_path, index=False)
        print(f"Top {TOP_K} genes saved to: {top_genes_path}")
    
    # Print summary table
    print("\n" + summary_df.to_string(index=False))
    
    return all_results


if __name__ == "__main__":
    results = main()