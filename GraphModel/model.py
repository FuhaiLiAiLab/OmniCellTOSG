import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax


import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax

from torch_geometric.nn import aggr
from torch_geometric.nn.conv import TransformerConv, GATConv, GCNConv, GINConv


class GraphDecoder(nn.Module):
    def __init__(self, model_name, input_dim, hidden_dim, embedding_dim, num_node, num_head, device, num_class):
        super(GraphDecoder, self).__init__()
        self.model_name = model_name
        self.num_node = num_node
        self.num_class = num_class
        self.num_head = num_head
        self.embedding_dim = embedding_dim

        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope=0.1)

        if model_name == 'gcn':
            num_head = 1
            self.num_head = num_head
            self.conv_first, self.conv_last = self.build_gcn_conv_layer(input_dim, hidden_dim, embedding_dim)
        elif model_name == 'gin':
            num_head = 1
            self.num_head = num_head
            self.conv_first, self.conv_last = self.build_gin_conv_layer(input_dim, hidden_dim, embedding_dim)
        elif model_name == 'gat':
            self.conv_first, self.conv_last = self.build_gat_conv_layer(input_dim, hidden_dim, embedding_dim)
        elif model_name == 'transformer':
            self.conv_first, self.conv_last = self.build_transformer_conv_layer(input_dim, hidden_dim, embedding_dim)
        else:
            raise ValueError('Unknown model name: {}'.format(model_name))
        
        self.z_norm = nn.BatchNorm1d(hidden_dim * num_head)
        self.x_norm_first = nn.BatchNorm1d(hidden_dim * num_head)
        self.x_norm_block = nn.BatchNorm1d(hidden_dim * num_head)
        self.x_norm_last = nn.BatchNorm1d(embedding_dim * num_head)
        self.internal_transformation = nn.Linear(hidden_dim * num_head, hidden_dim * num_head)
        # Simple aggregations
        self.mean_aggr = aggr.MeanAggregation()
        self.max_aggr = aggr.MaxAggregation()
        # Learnable aggregations
        self.softmax_aggr = aggr.SoftmaxAggregation(learn=True)
        self.powermean_aggr = aggr.PowerMeanAggregation(learn=True)
        # Graph prediction classifier
        self.graph_prediction = torch.nn.Linear(embedding_dim * self.num_head, num_class)

    def build_gcn_conv_layer(self, input_dim, hidden_dim, embedding_dim):
        conv_first = GCNConv(in_channels=input_dim, out_channels=hidden_dim)
        conv_last = GCNConv(in_channels=hidden_dim, out_channels=embedding_dim)
        return conv_first, conv_last
    
    def build_gin_conv_layer(self, input_dim, hidden_dim, embedding_dim):
        conv_first = GINConv(nn.Linear(input_dim, hidden_dim), train_eps=True)
        conv_last = GINConv(nn.Linear(hidden_dim, embedding_dim), train_eps=True)
        return conv_first, conv_last

    def build_gat_conv_layer(self, input_dim, hidden_dim, embedding_dim):
        conv_first = GATConv(in_channels=input_dim, out_channels=hidden_dim, heads=self.num_head)
        conv_last = GATConv(in_channels=hidden_dim * self.num_head, out_channels=embedding_dim, heads=self.num_head)
        return conv_first, conv_last

    def build_transformer_conv_layer(self, input_dim, hidden_dim, embedding_dim):
        conv_first = TransformerConv(in_channels=input_dim, out_channels=hidden_dim, heads=self.num_head)
        conv_last = TransformerConv(in_channels=hidden_dim * self.num_head, out_channels=embedding_dim, heads=self.num_head)
        return conv_first, conv_last
    
    def register_parameter(self):
        self.conv_first.reset_parameters()
        self.conv_last.reset_parameters()

    def forward(self, x, edge_index, internal_edge_index, ppi_edge_index, batch_size):
        # Internal message passing
        # import pdb; pdb.set_trace()
        z = self.conv_first(x, internal_edge_index)
        z = self.internal_transformation(z)
        z = self.z_norm(z)

        # Global message passing
        x = self.conv_first(x, edge_index) + z
        x = self.x_norm_first(x)
        x = self.act2(x)

        x = self.conv_last(x, edge_index)
        x = self.x_norm_last(x)
        x = self.act2(x)
        
        # Embedding decoder to [ypred]
        x = x.view(batch_size, self.num_node, self.embedding_dim * self.num_head)
        x = self.powermean_aggr(x).view(batch_size, self.embedding_dim * self.num_head)
        output = self.graph_prediction(x)
        _, ypred = torch.max(output, dim=1)
        return output, ypred

    def loss(self, output, label):
        num_class = self.num_class
        # Use weight vector to balance the loss
        weight_vector = torch.zeros([num_class]).to(device='cuda')
        label = label.long()
        for i in range(num_class):
            n_samplei = torch.sum(label == i)
            if n_samplei == 0:
                weight_vector[i] = 0
            else:
                weight_vector[i] = len(label) / (n_samplei)
        # Calculate the loss
        output = torch.log_softmax(output, dim=-1)
        loss = F.nll_loss(output, label, weight_vector)
        return loss