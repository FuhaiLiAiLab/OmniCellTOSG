import os
import re
import csv
import statistics
from datetime import datetime
from collections import defaultdict
import pandas as pd

ROOT_FOLDERS = ['Baseline_model_results', 'CellTOSG_model_results']
OUTPUT_DIR = 'model_results_sum'
SUMMARY_XLSX_PATH = os.path.join(OUTPUT_DIR, 'model_summary.xlsx')

# Mapping
DISEASE_NAME_MAP = {
    # 'acute_myeloid_leukemia': 'AML',
    'Alzheimers_Disease': 'AD',
    # 'clear_cell_renal_carcinoma': 'CCRC',
    # 'glioblastoma': 'GBM',
    "Lung_Adenocarcinoma": "LUAD",
    "Crohn_disease": "Crohn",
    "Lupus_Erythematosus,_Systemic": "SLE",

}

TASK_NAME_MAP = {
    'disease': 'Disease',
    'cell_type': 'Cell Type',
    'gender': 'Gender'
}

BASELINE_MODELS = ['dnn', 'scgpt', 'gat', 'gcn', 'transformer']
CELLTOSG_MODELS = ['gat', 'gcn', 'transformer']

TEST_ACC_PATTERN = re.compile(r'BEST MODEL TEST ACCURACY:\s*([0-9.]+)')

# Extract test accuracy
def extract_test_accuracy(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        for line in f:
            match = TEST_ACC_PATTERN.search(line)
            if match:
                return float(match.group(1))
    return None

# Format mean ± std string
def format_stat(values):
    if not values:
        return ""
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"

# Main processing
def process_all_results():
    combined_results = defaultdict(lambda: defaultdict(dict))  # (task, disease)[model][group] = [accs]

    for root_folder in ROOT_FOLDERS:
        group_name = 'Baseline' if 'Baseline' in root_folder else 'CellTOSG'
        for downstream_task in os.listdir(root_folder):
            if downstream_task not in TASK_NAME_MAP:
                continue
            for disease_dir in os.listdir(os.path.join(root_folder, downstream_task)):
                if disease_dir not in DISEASE_NAME_MAP:
                    continue
                task_path = os.path.join(root_folder, downstream_task, disease_dir)
                if not os.path.isdir(task_path):
                    continue

                model_acc_dict = defaultdict(list)
                for model_name in os.listdir(task_path):
                    model_path = os.path.join(task_path, model_name)
                    if not os.path.isdir(model_path):
                        continue

                    for folder in os.listdir(model_path):
                        full_path = os.path.join(model_path, folder, 'best_model_info.txt')
                        acc = extract_test_accuracy(full_path)
                        if acc is not None:
                            model_acc_dict[model_name.lower()].append(acc)

                # Save small CSV
                output_csv_dir = os.path.join(OUTPUT_DIR, downstream_task, disease_dir)
                os.makedirs(output_csv_dir, exist_ok=True)
                output_csv_path = os.path.join(output_csv_dir, f'{root_folder}.csv')
                with open(output_csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['model_name', 'test_accuracy (mean ± std)'])
                    for model, accs in sorted(model_acc_dict.items()):
                        writer.writerow([model, format_stat(accs)])

                print(f"Saved summary table: {output_csv_path}")

                # Save to combined structure
                task_display = TASK_NAME_MAP[downstream_task]
                disease_display = DISEASE_NAME_MAP[disease_dir]
                for model, accs in model_acc_dict.items():
                    combined_results[(task_display, disease_display)][model][group_name] = format_stat(accs)

    return combined_results

# Build and save combined summary table as XLSX
def build_summary_excel(combined_results):
    all_tasks = ['Disease', 'Cell Type', 'Gender']
    all_diseases = ['AD', 'LUAD', 'Crohn', 'SLE']
    baseline_cols = ['dnn', 'scgpt', 'gat', 'gcn', 'transformer']
    celltosg_cols = ['gat', 'gcn', 'transformer']

    multi_index = []
    for task in all_tasks:
        for disease in all_diseases:
            multi_index.append((task, disease))

    rows = []
    for task, disease in multi_index:
        row = []
        data = combined_results.get((task, disease), {})
        for model in baseline_cols:
            row.append(data.get(model, {}).get('Baseline', ''))
        for model in celltosg_cols:
            row.append(data.get(model, {}).get('CellTOSG', ''))
        rows.append(row)

    columns = (
        ['Baseline: ' + m.upper() for m in baseline_cols] +
        ['CellTOSG: ' + m.upper() for m in celltosg_cols]
    )
    df = pd.DataFrame(rows, columns=columns, index=pd.MultiIndex.from_tuples(multi_index, names=["Downstream Task", "Disease"]))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_excel(SUMMARY_XLSX_PATH)
    print(f"\n✅ Combined summary saved to {SUMMARY_XLSX_PATH}")

if __name__ == "__main__":
    results = process_all_results()
    build_summary_excel(results)
