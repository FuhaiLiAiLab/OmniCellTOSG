import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.inits import zeros

from torch_geometric.nn import aggr

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # X: [N, in_channels]
        # edge_index: [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        # [row] FOR 1st LINE && [col] FOR 2nd LINE
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-1/2)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

         # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # [aggr_out] OUT PUT DIMS = [N, out_channels]
        # import pdb; pdb.set_trace()
        return aggr_out


class GCNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, node_num, device, num_class):
        super(GCNDecoder, self).__init__()
        self.num_class = num_class
        self.node_num = node_num
        self.embedding_dim = embedding_dim
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(
                    input_dim, hidden_dim, embedding_dim)
        
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope=0.1)

        self.x_norm_first = nn.BatchNorm1d(hidden_dim)
        self.x_norm_block = nn.BatchNorm1d(hidden_dim)
        self.x_norm_last = nn.BatchNorm1d(embedding_dim)

        # Simple aggregations
        self.mean_aggr = aggr.MeanAggregation()
        self.max_aggr = aggr.MaxAggregation()
        # Learnable aggregations
        self.softmax_aggr = aggr.SoftmaxAggregation(learn=True)
        self.powermean_aggr = aggr.PowerMeanAggregation(learn=True)

        self.graph_prediction = torch.nn.Linear(embedding_dim, num_class)

    def build_conv_layer(self, input_dim, hidden_dim, embedding_dim):
        conv_first = GCNConv(in_channels=input_dim, out_channels=hidden_dim)
        conv_block = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        conv_last = GCNConv(in_channels=hidden_dim, out_channels=embedding_dim)
        return conv_first, conv_block, conv_last

    def forward(self, x, edge_index):
        # import pdb; pdb.set_trace()
        x = self.conv_first(x, edge_index)
        x = self.x_norm_first(x)
        x = self.act2(x)

        x = self.conv_block(x, edge_index)
        x = self.x_norm_block(x)
        x = self.act2(x)

        x = self.conv_last(x, edge_index)
        x = self.x_norm_last(x)
        x = self.act2(x)

        # Embedding decoder to [ypred]
        x = x.view(-1, self.node_num, self.embedding_dim)
        x = self.powermean_aggr(x).view(-1, self.embedding_dim)
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