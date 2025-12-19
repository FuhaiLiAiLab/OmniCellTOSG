from CellTOSG_Loader_new import CellTOSGDataLoader, CellTOSGSubsetBuilder
import numpy as np
import pandas as pd
import os
from difflib import get_close_matches

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


def explore_available_conditions(query_builder, fields=None, max_uniques=50):
    """
    Explore available conditions in the dataset.
    
    Parameters
    ----------
    query_builder : CellTOSGSubsetBuilder
        The subset builder instance
    fields : list, optional
        List of fields to explore. If None, uses default fields.
    max_uniques : int, optional
        Maximum number of unique values to return per field
        
    Returns
    -------
    dict : Dictionary containing available conditions
    """
    result = query_builder.available_conditions(
        include_fields=fields,
        max_uniques=max_uniques
    )
    return result


def soft_match_condition(query_builder, field, query_value, cutoff=0.6, n_matches=5):
    """
    Find closest matching values for a given field using fuzzy matching.
    Useful when exact match fails.
    
    Uses difflib.SequenceMatcher which finds the longest contiguous matching 
    subsequence. Good for typos and similar strings, but may miss abbreviations
    like "PDAC" -> "pancreatic ductal adenocarcinoma".
    
    Parameters
    ----------
    query_builder : CellTOSGSubsetBuilder
        The subset builder instance
    field : str
        Field name to search (e.g., 'disease', 'cell_type')
    query_value : str
        The value to search for
    cutoff : float
        Minimum similarity ratio (0-1) for matches
    n_matches : int
        Maximum number of matches to return
        
    Returns
    -------
    list : List of closest matching values
    """
    # Resolve field alias
    resolved_field = query_builder.FIELD_ALIAS.get(field, field)
    
    # Get available values for this field
    available = query_builder.available_conditions(
        include_fields=[resolved_field],
        max_uniques=None  # Get all values
    )
    
    if resolved_field not in available["unique_values"]:
        print(f"Field '{field}' (resolved: '{resolved_field}') not found in dataset.")
        return []
    
    all_values = available["unique_values"][resolved_field]["values"]
    
    # Convert to strings for matching
    all_values_str = [str(v) for v in all_values]
    
    # Find closest matches using SequenceMatcher (difflib)
    matches = get_close_matches(query_value.lower(), 
                                 [v.lower() for v in all_values_str], 
                                 n=n_matches, cutoff=cutoff)
    
    # Map back to original case
    matches_original = []
    for match in matches:
        for orig in all_values_str:
            if orig.lower() == match:
                matches_original.append(orig)
                break
    
    # Also do substring matching for cases like "PDAC" or partial terms
    query_lower = query_value.lower()
    substring_matches = [v for v in all_values_str 
                         if query_lower in v.lower() or v.lower() in query_lower]
    
    # Combine and deduplicate
    combined = matches_original + [m for m in substring_matches if m not in matches_original]
    
    return combined[:n_matches]


def print_available_conditions_summary(available_conds):
    """Pretty print the available conditions."""
    print("\n" + "="*80)
    print("AVAILABLE CONDITIONS IN DATASET")
    print("="*80)
    
    print(f"\nTotal metadata rows: {available_conds['metadata_rows']}")
    print(f"\nField aliases (use these in conditions):")
    for alias, actual in available_conds['field_alias'].items():
        print(f"  '{alias}' -> '{actual}'")
    
    print(f"\nUnique values per field:")
    for field, info in available_conds['unique_values'].items():
        print(f"\n  {field}:")
        print(f"    Total unique: {info['n_unique']}, Showing: {info['n_returned']}")
        # Show first 10 values as preview
        preview = info['values'][:10]
        print(f"    Preview: {preview}")
        if info['n_unique'] > 10:
            print(f"    ... and {info['n_unique'] - 10} more")


def test_available_conditions():
    """Test the available_conditions() functionality."""
    print("\n" + "="*80)
    print("TESTING available_conditions() FUNCTIONALITY")
    print("="*80)
    
    # Create a subset builder to explore conditions
    query_builder = CellTOSGSubsetBuilder(root=data_root)
    
    # 1. Get all available conditions
    print("\n--- 1. Exploring all available conditions ---")
    available = explore_available_conditions(query_builder, max_uniques=20)
    print_available_conditions_summary(available)
    
    # 2. Test soft matching for diseases
    print("\n--- 2. Testing soft matching for diseases ---")
    test_queries = [
        ("disease", "Alzheimer's Disease"),  # Note the apostrophe - might not exact match
        ("disease", "alzheimer"),             # Lowercase
        ("disease", "PDAC"),                  # Abbreviation
        ("disease", "lung cancer"),           # Generic term
        ("cell_type", "T cell"),              # Partial match
        ("cell_type", "ductal"),              # Partial match
    ]
    
    for field, query in test_queries:
        print(f"\n  Query: '{query}' in field '{field}'")
        matches = soft_match_condition(query_builder, field, query, cutoff=0.4, n_matches=5)
        if matches:
            print(f"  Closest matches: {matches}")
        else:
            print(f"  No matches found with cutoff 0.4")
    
    # 3. Save available conditions to CSV for reference
    print("\n--- 3. Saving available conditions to CSV ---")
    for field, info in available['unique_values'].items():
        field_df = pd.DataFrame({
            'value': info['values'],
            'field': field
        })
        csv_path = os.path.join(output_dir, f"available_{field}.csv")
        field_df.to_csv(csv_path, index=False)
        print(f"  Saved {field} values to: {csv_path}")
    
    return available, query_builder


# Global query builder for soft matching (initialized once)
_query_builder = None

def get_query_builder():
    """Get or create the global query builder instance."""
    global _query_builder
    if _query_builder is None:
        _query_builder = CellTOSGSubsetBuilder(root=data_root)
    return _query_builder


def test_retrieval_with_fallback(conditions, task, label_column, test_name, enable_soft_match=True):
    """
    Test data retrieval with given conditions.
    If hard match returns 0 samples and enable_soft_match=True, 
    automatically tries soft matching to find similar values.
    """
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"Conditions: {conditions}")
    print(f"{'='*60}")
    
    # First try hard match
    result, X, Y, metadata = _do_retrieval(conditions, task, label_column, test_name)
    
    # Check if we should try soft matching:
    # - enable_soft_match is True AND
    # - Either: num_samples == 0 with success, OR needs_soft_match flag is set (NO_SUBSET_RETRIEVED error)
    should_soft_match = enable_soft_match and (
        (result["num_samples"] == 0 and result["success"]) or 
        result.get("needs_soft_match", False)
    )
    
    if should_soft_match:
        print(f"\n⚠ Hard match returned 0 samples. Trying soft match...")
        
        query_builder = get_query_builder()
        new_conditions = conditions.copy()
        soft_matched = False
        
        for field, value in conditions.items():
            matches = soft_match_condition(query_builder, field, value, cutoff=0.3, n_matches=3)
            if matches and matches[0].lower() != value.lower():
                print(f"  Soft match for '{field}': '{value}' -> '{matches[0]}'")
                if len(matches) > 1:
                    print(f"    Other candidates: {matches[1:]}")
                new_conditions[field] = matches[0]
                soft_matched = True
        
        if soft_matched:
            print(f"\n  Retrying with soft-matched conditions: {new_conditions}")
            result, X, Y, metadata = _do_retrieval(
                new_conditions, task, label_column, 
                test_name + "_softmatched"
            )
            result["original_conditions"] = str(conditions)
            result["soft_matched"] = True
            result["soft_matched_conditions"] = str(new_conditions)
        else:
            print(f"  No better soft matches found.")
            result["soft_matched"] = False
    else:
        result["soft_matched"] = False
    
    return result, X, Y, metadata


def _do_retrieval(conditions, task, label_column, test_name, stratified_balancing=True):
    """
    Internal function to perform actual data retrieval.
    
    Parameters
    ----------
    stratified_balancing : bool, default=True
        If True, performs case-control balancing (matches disease samples with 
        healthy controls by cell type, sex, age). 
        If False, simply retrieves all samples matching the conditions.
    """
    try:
        dataset = CellTOSGDataLoader(
            root=data_root,
            conditions=conditions,
            task=task,
            label_column=label_column,
            sample_ratio=1.0,
            sample_size=None,
            shuffle=False,
            stratified_balancing=stratified_balancing,
            extract_mode="inference",
            random_state=2025,
            train_text=False,
            train_bio=False,
            dataset_correction=None,
            output_dir=os.path.join(output_dir, test_name.replace(" ", "_").replace(",", ""))
        )
        
        X, Y, metadata = dataset.data, dataset.labels, dataset.metadata
        
        # Handle empty results
        num_samples = 0
        num_features = 0
        if X is not None and hasattr(X, 'shape'):
            if len(X.shape) >= 1:
                num_samples = X.shape[0]
            if len(X.shape) >= 2:
                num_features = X.shape[1]
        
        result = {
            "test_name": test_name,
            "conditions": str(conditions),
            "num_samples": num_samples,
            "num_features": num_features,
            "success": True,
            "error": None,
            "needs_soft_match": num_samples == 0  # Flag for soft match fallback
        }
        
        print(f"✓ Retrieved {result['num_samples']} samples with {result['num_features']} features")
        
        # Get top K genes by mean expression
        if X is not None and num_samples > 0:
            mean_expr = np.mean(X, axis=0)
            top_k_indices = np.argsort(mean_expr)[-TOP_K:][::-1]
            top_k_values = mean_expr[top_k_indices]
            
            result["top_k_gene_indices"] = top_k_indices.tolist()
            result["top_k_gene_values"] = top_k_values.tolist()
            
            print(f"✓ Top {TOP_K} genes by mean expression computed")
            
        return result, X, Y, metadata
        
    except Exception as e:
        error_str = str(e)
        # Check if it's a NO_SUBSET_RETRIEVED error (needs soft match)
        needs_soft_match = "NO_SUBSET_RETRIEVED" in error_str
        print(f"✗ Error: {error_str}")
        return {
            "test_name": test_name,
            "conditions": str(conditions),
            "num_samples": 0,
            "num_features": 0,
            "success": False,
            "error": error_str,
            "needs_soft_match": needs_soft_match  # Flag for soft match fallback
        }, None, None, None


def test_retrieval(conditions, task, label_column, test_name):
    """Test data retrieval with given conditions (wrapper for backward compatibility)."""
    return test_retrieval_with_fallback(conditions, task, label_column, test_name, enable_soft_match=True)


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
    # First, test available_conditions() to see what's in the dataset
    available, query_builder = test_available_conditions()
    
    # Then run the retrieval tests
    results = main()