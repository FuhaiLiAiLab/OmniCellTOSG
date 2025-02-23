import torch
import numpy as np
import networkx as nx
import torch.utils.data

from torch_geometric.data import Batch
from torch_geometric.data import DataListLoader

class GeoGraphLoader():
    def __init__(self):
        pass

    def load_graph(geo_datalist, batch_size, num_workers):
        dataset_sampler = GeoGraphSampler(geo_datalist)
        dataset_loader = torch.utils.data.DataLoader(
                                        dataset=dataset_sampler, 
                                        batch_size=batch_size, 
                                        shuffle=False,
                                        num_workers=num_workers,
                                        collate_fn=collate_fn)
        return dataset_loader

# Custom graph dataset with [init, len, getitem]
class GeoGraphSampler(torch.utils.data.Dataset):
    # Initiate with [node_num, feature_dim, length, geo_batch_datalist] from [geo_datalist]
    def __init__(self, geo_datalist):
        self.node_num = geo_datalist[0].x.shape[0]
        self.feature_dim = geo_datalist[0].x.shape[1]
        self.length = len(geo_datalist)
        self.geo_batch_datalist = Batch.from_data_list(geo_datalist)

    def __len__(self):
        return self.length

    # Input point: a graph (all of nodes, and its nodes features)
    # Input label: a graph label
    def __getitem__(self, idx):
        return self.geo_batch_datalist[idx]

def collate_fn(examples):
    data_list = Batch.from_data_list(examples)
    return (data_list)