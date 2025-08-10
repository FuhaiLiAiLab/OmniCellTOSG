import os
import csv
import re
from datetime import datetime
from collections import defaultdict

# set path
ROOT_FOLDERS = ['Baseline_model_results', 'CellTOSG_model_results']
OUTPUT_DIR = 'model_results_sum'

# test accuracy field
TEST_ACC_PATTERN = re.compile(r'BEST MODEL TEST ACCURACY:\s*([0-9.]+)')

# get latest timestamp folder
def get_latest_timestamp_folder(model_path):
    timestamp_folders = []
    for folder in os.listdir(model_path):
        if os.path.isdir(os.path.join(model_path, folder)):
            match = re.search(r'\d{8}_\d{6}', folder)
            if match:
                timestamp = match.group()
                timestamp_dt = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                timestamp_folders.append((timestamp_dt, folder))
    if not timestamp_folders:
        return None
    # return the folder with the latest timestamp
    latest_dt, latest_folder = max(timestamp_folders, key=lambda x: x[0])
    print(f"[INFO] Latest timestamp folder: {latest_folder} (Datetime: {latest_dt.strftime('%Y-%m-%d %H:%M:%S')})")

    return os.path.join(model_path, latest_folder)

# extract test accuracy
def extract_test_accuracy(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            match = TEST_ACC_PATTERN.search(line)
            if match:
                return float(match.group(1))
    return None

def process_all_results():
    for root_folder in ROOT_FOLDERS:
        for downstream_task in os.listdir(root_folder):
            downstream_path = os.path.join(root_folder, downstream_task)
            if not os.path.isdir(downstream_path):
                continue

            for disease_name in os.listdir(downstream_path):
                disease_path = os.path.join(downstream_path, disease_name)
                if not os.path.isdir(disease_path):
                    continue

                model_acc_dict = {}

                for model_name in os.listdir(disease_path):
                    model_path = os.path.join(disease_path, model_name)
                    if not os.path.isdir(model_path):
                        continue

                    latest_folder = get_latest_timestamp_folder(model_path)
                    if latest_folder:
                        best_model_info_path = os.path.join(latest_folder, 'best_model_info.txt')
                        test_acc = extract_test_accuracy(best_model_info_path)
                        if test_acc is not None:
                            model_acc_dict[model_name] = test_acc

                # output directory
                output_csv_dir = os.path.join(OUTPUT_DIR, downstream_task, disease_name)
                os.makedirs(output_csv_dir, exist_ok=True)

                output_csv_path = os.path.join(output_csv_dir, f'{root_folder}.csv')

                # write to CSV
                with open(output_csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['model_name', 'test_accuracy'])
                    for model, acc in sorted(model_acc_dict.items()):
                        writer.writerow([model, acc])

                print(f"Saved: {output_csv_path}")

if __name__ == "__main__":
    process_all_results()
