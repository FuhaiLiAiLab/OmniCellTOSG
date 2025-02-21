import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import aggr

# GAT torch_geometric implementation
# Adapted from https://github.com/snap-stanford/pretrain-gnns
class GATConv(MessagePassing):
    def __init__(self, input_dim, embed_dim, num_head=1, negative_slope=0.2, aggr="add", num_edge_type=0):
        super(GATConv, self).__init__(node_dim=0)
        assert embed_dim % num_head == 0
        self.k = embed_dim // num_head
        self.aggr = aggr

        self.embed_dim = embed_dim
        self.num_head = num_head
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(input_dim, embed_dim,bias=False)
        self.att = torch.nn.Parameter(torch.Tensor(1, num_head, 2 * self.k))
        self.bias = torch.nn.Parameter(torch.Tensor(embed_dim))

        if num_edge_type > 0:
            self.edge_embedding = torch.nn.Embedding(num_edge_type, embed_dim)
            nn.init.xavier_uniform_(self.edge_embedding.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_linear.weight.data)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index):
        # import pdb; pdb.set_trace()
        #add self loops in the edge space
        edge_index,_ = add_self_loops(edge_index, num_nodes = x.size(0))
        x = self.weight_linear(x).view(-1, self.num_head, self.k) # N * num_head * k

        return self.propagate(edge_index, x=x)

    def message(self, edge_index, x_i, x_j):
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1) # E * num_head
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])
        return x_j * alpha.view(-1, self.num_head, 1) #E * num_head * k

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1,self.embed_dim)
        aggr_out = aggr_out + self.bias
        return F.relu(aggr_out)


class GATDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, node_num, num_head, device, num_class):
        super(GATDecoder, self).__init__()
        self.num_class = num_class
        self.node_num = node_num
        self.num_head = num_head
        self.embedding_dim = embedding_dim
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(
                    input_dim, hidden_dim, embedding_dim)
        
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope=0.2)

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
        conv_first = GATConv(input_dim=input_dim, embed_dim=hidden_dim, num_head=self.num_head)
        conv_block = GATConv(input_dim=hidden_dim, embed_dim=hidden_dim, num_head=self.num_head)
        conv_last = GATConv(input_dim=hidden_dim, embed_dim=embedding_dim, num_head=self.num_head)
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