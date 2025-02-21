import os
import numpy as np
import pandas as pd

x_file_path = './CellTOG/brain_sc_output/processed_data/brain/alzheimer\'s_disease/alzheimer\'s_disease_X_partition_0.npy'
y_file_path = './CellTOG/brain_sc_output/processed_data/brain/alzheimer\'s_disease/alzheimer\'s_disease_Y_partition_0.npy'

x = np.load(x_file_path)
y = np.load(y_file_path)
print(x.shape, y.shape)
