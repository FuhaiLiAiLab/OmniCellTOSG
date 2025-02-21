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

    def form_geo_datalist(self, num_graph, graph_feature_list, graph_label_list, all_edge_index, internal_edge_index, ppi_edge_index):
        geo_datalist = []
        for i in range(num_graph):
            graph_feature = graph_feature_list[i]
            graph_label = graph_label_list[i]
            # CONVERT [numpy] TO [torch]
            graph_feature = torch.from_numpy(graph_feature).float()
            graph_label = torch.from_numpy(np.array([graph_label])).float()
            geo_data = Data(x=graph_feature, edge_index=ppi_edge_index, internal_edge_index=internal_edge_index, all_edge_index=all_edge_index, label=graph_label)
            geo_datalist.append(geo_data)
        return geo_datalist


def read_batch(index, upper_index, x_input, y_input, num_feature, num_node, all_edge_index, internal_edge_index, ppi_edge_index, graph_output_folder):
    # FORMING BATCH FILES
    form_data_path = './' + graph_output_folder + '/form_data'
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

