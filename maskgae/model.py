import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import (
    Linear,
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    GATv2Conv,
    global_add_pool,
    global_mean_pool,
    global_max_pool
)

from torch_geometric.utils import add_self_loops, negative_sampling, degree
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn.inits import glorot, zeros

import math
from tqdm.auto import tqdm
from typing import Optional, Tuple, Union
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax


# custom modules
from maskgae.loss import info_nce_loss, ce_loss, log_rank_loss, hinge_auc_loss, auc_loss


def to_sparse_tensor(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)


def creat_gnn_layer(name, first_channels, second_channels, heads):
    if name == "sage":
        layer = SAGEConv(first_channels, second_channels)
    elif name == "gcn":
        layer = GCNConv(first_channels, second_channels)
    elif name == "gin":
        layer = GINConv(Linear(first_channels, second_channels), train_eps=True)
    elif name == "gat":
        layer = GATConv(-1, second_channels, heads=heads)
    elif name == "gat2":
        layer = GATv2Conv(-1, second_channels, heads=heads)
    else:
        raise ValueError(name)
    return layer


def create_input_layer(num_nodes, num_node_feats,
                       use_node_feats=True, node_emb=None):
    emb = None
    if use_node_feats:
        input_dim = num_node_feats
        if node_emb:
            emb = torch.nn.Embedding(num_nodes, node_emb)
            input_dim = input_dim + node_emb
    else:
        emb = torch.nn.Embedding(num_nodes, node_emb)
        input_dim = node_emb
    return input_dim, emb


def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU(inplace=False)
    elif activation == "elu":
        return nn.ELU(inplace=False)
    else:
        raise ValueError("Unknown activation")


class TransformerConv(MessagePassing):
    _alpha: OptTensor
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights=None,
    ):
        # forward_type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # forward_type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # forward_type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # forward_type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        bn=False,
        layer="gcn",
        activation="elu",
        use_node_feats=True,
        num_nodes=None,
        node_emb=None,
    ):

        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.use_node_feats = use_node_feats
        self.node_emb = node_emb

        if node_emb is not None and num_nodes is None:
            raise RuntimeError("Please provide the argument `num_nodes`.")

        in_channels, self.emb = create_input_layer(
            num_nodes, in_channels, use_node_feats=use_node_feats, node_emb=node_emb
        )
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            heads = 1 if i == num_layers - 1 or 'gat' not in layer else 4

            self.convs.append(creat_gnn_layer(layer, first_channels, second_channels, heads))
            self.bns.append(bn(second_channels*heads))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.bns:
            if not isinstance(bn, nn.Identity):
                bn.reset_parameters()

        if self.emb is not None:
            nn.init.xavier_uniform_(self.emb.weight)

    def create_input_feat(self, x):
        if self.use_node_feats:
            input_feat = x
            if self.node_emb:
                input_feat = torch.cat([self.emb.weight, input_feat], dim=-1)
        else:
            input_feat = self.emb.weight
        return input_feat

    def forward(self, x, edge_index):
        x = self.create_input_feat(x)
        edge_index = to_sparse_tensor(edge_index, x.size(0))

        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        return x

    @torch.no_grad()
    def get_embedding(self, x, edge_index, mode="cat"):

        self.eval()
        assert mode in {"cat", "last"}, mode

        x = self.create_input_feat(x)
        edge_index = to_sparse_tensor(edge_index, x.size(0))
        out = []
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            out.append(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        out.append(x)

        if mode == "cat":
            embedding = torch.cat(out, dim=1)
        else:
            embedding = out[-1]

        return embedding


class DotEdgeDecoder(nn.Module):
    """Simple Dot Product Edge Decoder"""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def reset_parameters(self):
        return

    def forward(self, z, edge, sigmoid=True):
        x = z[edge[0]] * z[edge[1]]
        x = x.sum(-1)

        if sigmoid:
            return x.sigmoid()
        else:
            return x


class EdgeDecoder(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
        self, in_channels, hidden_channels, out_channels=1,
        num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, z, edge, sigmoid=True, reduction=False):
        x = z[edge[0]] * z[edge[1]]

        if reduction:
            x = x.mean(1)

        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)

        x = self.mlps[-1](x)

        if sigmoid:
            return x.sigmoid()
        else:
            return x


class DegreeDecoder(nn.Module):
    """Simple MLP Degree Decoder"""

    def __init__(
        self, in_channels, hidden_channels, out_channels=1,
        num_layers=2, dropout=0.5, activation='relu',
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels, second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, x):

        for i, mlp in enumerate(self.mlps[:-1]):
            x = mlp(x)
            x = self.dropout(x)
            x = self.activation(x)

        x = self.mlps[-1](x)
        x = self.activation(x)

        return x


def random_negative_sampler(edge_index, num_nodes, num_neg_samples):
    neg_edges = torch.randint(0, num_nodes, size=(2, num_neg_samples)).to(edge_index)
    return neg_edges


class MaskGAE(nn.Module):
    def __init__(
        self,
        input_dim,
        num_node,
        encoder,
        internal_encoder,
        edge_decoder,
        degree_decoder=None,
        mask=None,
        random_negative_sampling=False,
        loss="ce",
    ):
        super().__init__()
        self.encoder = encoder
        self.edge_decoder = edge_decoder
        self.internal_encoder = internal_encoder
        self.degree_decoder = degree_decoder
        self.mask = mask

        self.num_node = num_node

        # self.att_linear = torch.nn.Linear(input_dim * 2, input_dim)

        if loss == "ce":
            self.loss_fn = ce_loss
        elif loss == "auc":
            self.loss_fn = auc_loss
        elif loss == "info_nce":
            self.loss_fn = info_nce_loss
        elif loss == "log_rank":
            self.loss_fn = log_rank_loss
        elif loss == "hinge_auc":
            self.loss_fn = hinge_auc_loss
        else:
            raise ValueError(loss)

        if random_negative_sampling:
            # this will be faster than pyg negative_sampling
            self.negative_sampler = random_negative_sampler
        else:
            self.negative_sampler = negative_sampling

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.internal_encoder.reset_parameters()
        self.edge_decoder.reset_parameters()

        if self.degree_decoder is not None:
            self.degree_decoder.reset_parameters()

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

    def train_step(self, data, optimizer, alpha=0.002,
        batch_size=2 ** 16, grad_norm=1.0
    ):
        self.train()

        num_node = self.num_node * batch_size
        x, edge_index = data.x, data.edge_index
        internal_edge_index = data.internal_edge_index

        if self.mask is not None:
            # MaskGAE
            remaining_edges, masked_edges = self.mask(edge_index)
        else:
            # Plain GAE
            remaining_edges = edge_index
            masked_edges = getattr(data, "pos_edge_label_index", edge_index)

        batch_count = 0
        loss_total = 0.0
        aug_edge_index, _ = add_self_loops(edge_index)
        neg_edges = self.negative_sampler(
            aug_edge_index,
            num_nodes=num_node,
            num_neg_samples=masked_edges.view(2, -1).size(1),
        ).view_as(masked_edges)

        # Create a tqdm progress bar
        progress_bar = tqdm(DataLoader(
            range(masked_edges.size(1)), batch_size=batch_size, shuffle=True
        ), desc="Processing Batches")

        for perm in progress_bar:

            optimizer.zero_grad()

            ### BUILD UP ASSIGNMENT MATRIX

            # x_norm = self.x_norm(x)
            # x_norm = x_norm.reshape(-1, self.node_num, self.input_dim)
            # x = self.gcn(x, internal_edge_index)

            z = self.internal_encoder(x, internal_edge_index)
            # x = self.att_linear(x)


            z = self.encoder(z, remaining_edges)

            batch_masked_edges = masked_edges[:, perm]
            batch_neg_edges = neg_edges[:, perm]

            # ******************* loss for edge reconstruction *********************
            pos_out = self.edge_decoder(
                z, batch_masked_edges, sigmoid=False
            )
            neg_out = self.edge_decoder(z, batch_neg_edges, sigmoid=False)
            loss = self.loss_fn(pos_out, neg_out)
            # **********************************************************************

            # ******************* loss for degree prediction ***********************
            if self.degree_decoder is not None and alpha:
                deg = degree(masked_edges[1].flatten(), num_node).float()
                loss = loss + alpha * F.mse_loss(self.degree_decoder(z).squeeze(), deg)
            # **********************************************************************

            torch.autograd.set_detect_anomaly(True)
            with torch.autograd.detect_anomaly():
                # loss.backward() 
                loss.backward(retain_graph=True) 

            if grad_norm > 0:
                # gradient clipping
                nn.utils.clip_grad_norm_(self.parameters(), grad_norm)

            optimizer.step()

            loss_total = loss_total + loss.item()
            batch_count = batch_count + 1

            avg_loss = loss_total / batch_count
            progress_bar.set_postfix(avg_loss=avg_loss)

        return loss_total

    @torch.no_grad()
    def batch_predict(self, z, edges, batch_size=2 ** 16):
        preds = []
        for perm in DataLoader(range(edges.size(1)), batch_size):
            edge = edges[:, perm]
            preds = preds + [self.edge_decoder(z, edge).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred

    @torch.no_grad()
    def test_step(self, data, pos_edge_index, neg_edge_index, batch_size=2**16):
        self.eval()
        num_node = self.num_node * batch_size
        x, edge_index = data.x, data.edge_index
        internal_edge_index = data.internal_edge_index

        z = self.internal_encoder(x, internal_edge_index)
        z = self(z, edge_index)
        pos_pred = self.batch_predict(z, pos_edge_index)
        neg_pred = self.batch_predict(z, neg_edge_index)

        pred = torch.cat([pos_pred, neg_pred], dim=0)
        pos_y = pos_pred.new_ones(pos_pred.size(0))
        neg_y = neg_pred.new_zeros(neg_pred.size(0))

        y = torch.cat([pos_y, neg_y], dim=0)
        y, pred = y.cpu().numpy(), pred.cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)

    @torch.no_grad()
    def test_step_ogb(self, data, evaluator, 
                      pos_edge_index, neg_edge_index, batch_size=2**16):
        self.eval()
        z = self(data.x, data.edge_index)
        pos_pred = self.batch_predict(z, pos_edge_index)
        neg_pred = self.batch_predict(z, neg_edge_index)

        results = {}
        for K in [20, 50, 100]:
            evaluator.K = K
            hits = evaluator.eval(
                {"y_pred_pos": pos_pred, "y_pred_neg": neg_pred, }
            )[f"hits@{K}"]
            results[f"Hits@{K}"] = hits

        return results
