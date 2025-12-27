import json
import torch
import numpy as np
import pandas as pd
import networkx as nx

from numpy import inf
from torch_geometric.data import Data

class ReadGeoGraph():
    def __init__(self):
        pass

    def read_feature(self, num_graph, num_feature, num_node, xBatch):
        # FORM [graph_feature_list]
        xBatch = xBatch.reshape(num_graph, num_node, num_feature)
        graph_feature_list = []
        for i in range(num_graph):
            graph_feature_list.append(xBatch[i, :, :])
        return graph_feature_list

    def read_label(self, yBatch):
        yBatch_list = [label[0] for label in list(yBatch)]
        graph_label_list = yBatch_list
        return graph_label_list

    def form_pretrain_geo_datalist(self, num_graph, graph_feature_list, all_edge_index, internal_edge_index, ppi_edge_index):
        pretrain_geo_datalist = []
        for i in range(num_graph):
            graph_feature = graph_feature_list[i]
            pretrain_geo_data = Data(x=graph_feature, edge_index=ppi_edge_index, internal_edge_index=internal_edge_index, all_edge_index=all_edge_index)
            pretrain_geo_datalist.append(pretrain_geo_data)
        return pretrain_geo_datalist

    def form_geo_datalist(self, num_graph, graph_feature_list, graph_label_list, all_edge_index, internal_edge_index, ppi_edge_index):
        geo_datalist = []
        for i in range(num_graph):
            graph_feature = graph_feature_list[i]
            graph_label = graph_label_list[i]
            geo_data = Data(x=graph_feature, edge_index=ppi_edge_index, internal_edge_index=internal_edge_index, all_edge_index=all_edge_index, label=graph_label)
            geo_datalist.append(geo_data)
        return geo_datalist


def read_pretrain_batch(index, upper_index, x_input, num_feature, num_node, all_edge_index, internal_edge_index, ppi_edge_index):
    # FORMING BATCH FILES
    print('--------------' + str(index) + ' to ' + str(upper_index) + '--------------')
    xBatch = x_input[index : upper_index, :]
    print(xBatch.shape)
    # PREPARE LOADING LISTS OF [features, labels, drugs, edge_index]
    print('READING BATCH GRAPHS TO LISTS ...')
    num_graph = upper_index - index
    # print('READING BATCH FEATURES ...')
    graph_feature_list =  ReadGeoGraph().read_feature(num_graph, num_feature, num_node, xBatch)
    # print('FORMING GEOMETRIC GRAPH DATALIST ...')
    pretrain_geo_datalist = ReadGeoGraph().form_pretrain_geo_datalist(num_graph, graph_feature_list, all_edge_index, internal_edge_index, ppi_edge_index)
    return pretrain_geo_datalist


def read_batch(index, upper_index, x_input, y_input, num_feature, num_node, all_edge_index, internal_edge_index, ppi_edge_index):
    # FORMING BATCH FILES
    print('--------------' + str(index) + ' to ' + str(upper_index) + '--------------')
    xBatch = x_input[index : upper_index, :]
    yBatch = y_input[index : upper_index, :]
    print(xBatch.shape)
    print(yBatch.shape)
    # PREPARE LOADING LISTS OF [features, labels, drugs, edge_index]
    print('READING BATCH GRAPHS TO LISTS ...')
    num_graph = upper_index - index
    # print('READING BATCH FEATURES ...')
    graph_feature_list =  ReadGeoGraph().read_feature(num_graph, num_feature, num_node, xBatch)
    # print('READING BATCH LABELS ...')
    graph_label_list = ReadGeoGraph().read_label(yBatch)
    # print('FORMING GEOMETRIC GRAPH DATALIST ...')
    geo_datalist = ReadGeoGraph().form_geo_datalist(num_graph, graph_feature_list, graph_label_list, all_edge_index, internal_edge_index, ppi_edge_index)
    return geo_datalist


def read_drug_batch(index, upper_index, drug_input):
    # FORMING BATCH FILES
    print('--------------' + str(index) + ' to ' + str(upper_index) + '--------------')
    drugBatch = drug_input[index : upper_index, :]
    # Flatten the array to 1D
    drugBatch = drugBatch.flatten()
    # import pdb; pdb.set_trace()
    print(drugBatch.shape)
    # Load drug database JSON file
    with open('/storage1/fs1/fuhai.li/Active/hemingzhang/OmniCellTOSG/drug_database.json', 'r') as f:
        drug_database = json.load(f)

    # Get list of drug names (should match indices 0-4)
    drug_names = list(drug_database.keys())
    print(f"Available drugs: {drug_names}")
    
    # Create list to store drug graphs for this batch
    drug_graph_list = []
    
    for drug_idx in drugBatch:
        # Get drug name from index
        drug_name = drug_names[drug_idx]
        drug_info = drug_database[drug_name]
        
        # Extract drug features
        smiles = drug_info['smiles']
        atom_features = torch.tensor(drug_info['atom_features'], dtype=torch.float)
        edge_index = torch.tensor(drug_info['edge_index'], dtype=torch.long)
        
        # Create PyG Data object for drug
        drug_data = Data(x=atom_features, edge_index=edge_index)
        drug_graph_list.append(drug_data)
        
        print(f"Loaded drug: {drug_name}, atoms: {atom_features.shape[0]}, edges: {edge_index.shape[1]}")
    
    return drug_graph_list