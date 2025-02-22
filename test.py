import os
import numpy as np
import pandas as pd
import torch

class PreGraph:
    def __init__(self, edge_index, num_entity):
        self.edge_index = edge_index
        self.num_entity = num_entity


# Read these feature label files
print('--- LOADING TRAINING FILES ... ---')
x_file_path = './CellTOG/brain_sc_output/processed_data/brain/alzheimer\'s_disease/alzheimer\'s_disease_X_partition_0.npy'
y_file_path = './CellTOG/brain_sc_output/processed_data/brain/alzheimer\'s_disease/alzheimer\'s_disease_Y_partition_0.npy'
xAll = np.load(x_file_path)
yAll = np.load(y_file_path)
print(xAll.shape, yAll.shape)

num_cell = xAll.shape[0]
yAll = yAll.reshape(num_cell, -1)

all_edge_index = torch.from_numpy(np.load('./CellTOG/edge_index.npy')).long()
internal_edge_index = torch.from_numpy(np.load('./CellTOG/internal_edge_index.npy')).long()
ppi_edge_index = torch.from_numpy(np.load('./CellTOG/ppi_edge_index.npy')).long()

num_entity = xAll.shape[1]

