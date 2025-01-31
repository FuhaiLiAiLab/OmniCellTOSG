import os
import pandas as pd

root_dir_path = './sc_meta_cell/output'
x_data_folder = root_dir_path + '/X_data'
y_data_folder = root_dir_path + '/Y_data'

x_file_path = os.path.join(x_data_folder, 'acute myeloid leukemia_bone marrow_partition_0_ENSG_id_processed.csv')
y_file_path = os.path.join(y_data_folder, 'acute myeloid leukemia_bone marrow_partition_0_purity.csv')

# Load expression data (X) and label data (Y)
x_data = pd.read_csv(x_file_path)  # Shape: [num_samples, num_genes]
y_data = pd.read_csv(y_file_path)  # Shape: [num_samples, cluster/label]

print(x_data)