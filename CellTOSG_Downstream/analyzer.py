import os
import pandas as pd
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
from torch_geometric.nn import aggr


def to_sparse_tensor(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)


def creat_gnn_layer(name, first_channels, second_channels, heads):
    if name == "gcn":
        layer = GCNConv(first_channels, second_channels)
    elif name == "gin":
        layer = GINConv(Linear(first_channels, second_channels), train_eps=True)
    elif name == "gat":
        layer = GATConv(-1, second_channels, heads=heads)
    elif name == "transformer":
        layer = TransformerConv(first_channels, second_channels, heads=heads)
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
    elif activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2, inplace=False)
    elif activation == "elu":
        return nn.ELU(inplace=False)
    else:
        raise ValueError("Unknown activation")


def create_linear_activation_layer(linear_activation):
    if linear_activation is None:
        return nn.Identity()
    elif linear_activation == "relu":
        return nn.ReLU(inplace=False)
    elif linear_activation == "elu":
        return nn.ELU(inplace=False)
    elif linear_activation == "leaky_relu":
        return nn.LeakyReLU(inplace=False)
    elif linear_activation == "gelu":
        return nn.GELU(inplace=False)
    else:
        raise ValueError("Unknown linear activation")


def split_attention_by_batch_and_layer(attention_weights, batch_size, num_entity, index_list, remove_self_loops=True):
    """
    Split batched attention weights by batch samples first, then by layers.
    
    Args:
        attention_weights: List of (edge_pairs, attention_vals) for each layer
        batch_size: number of samples in the batch
        num_entity: number of entities per sample
        index_list: list of batch indices [index, index+1, ..., index+batch_size-1]
        remove_self_loops: whether to remove self-loop edges
    
    Returns:
        Dictionary with structure: {batch_index: {layer_name: (edge_pairs, attention_vals)}}
    """
    # Define layer names based on number of layers
    num_layers = len(attention_weights)
    layer_names = []
    
    if num_layers == 1:
        layer_names = ['single_layer']
    elif num_layers == 2:
        layer_names = ['initial_layer', 'last_layer']
    else:
        layer_names = ['initial_layer']
        for i in range(1, num_layers - 1):
            layer_names.append(f'block_layer_{i}')
        layer_names.append('last_layer')
    
    # Initialize the batch-first structure
    batch_attention_analysis = {}
    for batch_idx, actual_index in enumerate(index_list):
        batch_attention_analysis[actual_index] = {}
    
    # Process each layer
    for layer_idx, (edge_pairs, attention_vals) in enumerate(attention_weights):
        layer_name = layer_names[layer_idx]
        
        # Process each batch sample
        for batch_idx, actual_index in enumerate(index_list):
            # Calculate node offset for this batch
            node_offset = batch_idx * num_entity
            node_start = node_offset
            node_end = node_offset + num_entity
            
            # Find edges that belong to this batch
            source_mask = (edge_pairs[0] >= node_start) & (edge_pairs[0] < node_end)
            target_mask = (edge_pairs[1] >= node_start) & (edge_pairs[1] < node_end)
            edge_mask = source_mask & target_mask
            
            # Extract edges for this batch
            batch_edge_pairs = edge_pairs[:, edge_mask]
            batch_attention_vals = attention_vals[edge_mask]
            
            # Convert back to local node indices (subtract offset)
            batch_edge_pairs = batch_edge_pairs - node_offset
            
            # Remove self-loops if requested
            if remove_self_loops and batch_edge_pairs.size(1) > 0:
                # Find non-self-loop edges (source != target)
                non_self_loop_mask = batch_edge_pairs[0] != batch_edge_pairs[1]
                batch_edge_pairs = batch_edge_pairs[:, non_self_loop_mask]
                batch_attention_vals = batch_attention_vals[non_self_loop_mask]
            
            # Store in batch-first structure
            batch_attention_analysis[actual_index][layer_name] = (batch_edge_pairs, batch_attention_vals)
    
    return batch_attention_analysis


def extract_attention_from_sparse_tensor(attention_info, average_heads=True):
    """Extract from, to pairs and attention values from SparseTensor."""
    if isinstance(attention_info, SparseTensor):
        # Get the COO format (coordinate format)
        row, col, val = attention_info.coo()
        
        # Create from-to pairs
        edge_pairs = torch.stack([row, col], dim=0)  # Shape: [2, num_edges]
        
        # Get attention values
        attention_values = val  # Shape: [num_edges, num_heads]
        
        # Average across heads if requested
        if average_heads and attention_values.dim() > 1:
            attention_values = attention_values.mean(dim=1)  # Shape: [num_edges]
        
        return edge_pairs, attention_values
    else:
        # Handle tuple format (edge_index, attention_weights)
        return attention_info


def save_attention_analysis_to_folders(batch_attention_analysis, output_dir="attention_analysis"):
    """
    Save batch attention analysis as folders with CSV files.
    
    Args:
        batch_attention_analysis: Dictionary with {batch_index: {layer_name: (edge_pairs, attention_vals)}}
        output_dir: Base directory to save the analysis
    """
    # Create the base output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for batch_index, layer_data in batch_attention_analysis.items():
        # Create folder for this sample
        sample_folder = os.path.join(output_dir, f"sample_{batch_index}")
        os.makedirs(sample_folder, exist_ok=True)
        
        for layer_name, (edge_pairs, attention_vals) in layer_data.items():
            # Convert tensors to numpy for easier handling
            if edge_pairs.size(1) > 0:  # Check if there are any edges
                from_nodes = edge_pairs[0].detach().cpu().numpy()
                to_nodes = edge_pairs[1].detach().cpu().numpy()
                
                # Handle attention values - squeeze if needed and detach
                if attention_vals.dim() > 1:
                    values = attention_vals.squeeze().detach().cpu().numpy()
                else:
                    values = attention_vals.detach().cpu().numpy()
                
                # Create DataFrame
                df = pd.DataFrame({
                    'from': from_nodes,
                    'to': to_nodes, 
                    'value': values
                })
            else:
                # Create empty DataFrame if no edges
                df = pd.DataFrame({
                    'from': [],
                    'to': [],
                    'value': []
                })
            
            # Save as CSV
            csv_filename = f"{layer_name}.csv"
            csv_path = os.path.join(sample_folder, csv_filename)
            df.to_csv(csv_path, index=False)
            
            print(f"Saved {csv_filename} for sample_{batch_index} with {len(df)} edges")


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


class DownGNNEncoder(nn.Module):
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

    # Modify your forward_with_attention method to handle this:
    def forward_with_attention(self, x, edge_index):
        """Forward pass that returns both output and attention weights for GAT layers."""
        x = self.create_input_feat(x)
        edge_index = to_sparse_tensor(edge_index, x.size(0))
        
        attention_weights = []
        
        # Process all layers except the last one
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            if 'gat' in str(type(conv)).lower():
                # For GAT layers, extract attention weights
                result = conv(x, edge_index, return_attention_weights=True)
                x, attention_info = result
                # Convert SparseTensor to edge pairs and values
                edge_pairs, attention_vals = extract_attention_from_sparse_tensor(attention_info)
                attention_weights.append((edge_pairs, attention_vals))
            else:
                x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
        
        # Process the last layer
        x = self.dropout(x)
        if 'gat' in str(type(self.convs[-1])).lower():
            result = self.convs[-1](x, edge_index, return_attention_weights=True)
            x, attention_info = result
            # Convert SparseTensor to edge pairs and values
            edge_pairs, attention_vals = extract_attention_from_sparse_tensor(attention_info)
            attention_weights.append((edge_pairs, attention_vals))
        else:
            x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        
        return x, attention_weights
    

class CellTOSG_Class(nn.Module):
    def __init__(
        self,
        text_input_dim,
        omic_input_dim,
        cross_fusion_output_dim,
        pre_input_output_dim,
        final_fusion_output_dim,
        num_class,
        num_entity,
        linear_hidden_dims,
        linear_activation,
        linear_dropout_rate,
        text_encoder,
        rna_seq_encoder,
        prot_seq_encoder,
        encoder,
        internal_encoder
    ):
        super().__init__()

        self.num_class = num_class

        self.text_encoder = text_encoder
        self.rna_seq_encoder = rna_seq_encoder
        self.prot_seq_encoder = prot_seq_encoder
        self.encoder = encoder
        self.internal_encoder = internal_encoder

        self.name_linear_transform = nn.Linear(text_input_dim, text_input_dim)
        self.desc_linear_transform = nn.Linear(text_input_dim, text_input_dim)
        self.seq_linear_transform = nn.Linear(text_input_dim, text_input_dim)
        self.omic_linear_transform = nn.Linear(omic_input_dim, omic_input_dim)
        self.cross_modal_fusion = nn.Linear(text_input_dim * 3 + omic_input_dim, cross_fusion_output_dim)
        self.pre_transform = nn.Linear(pre_input_output_dim, pre_input_output_dim)
        self.fusion = nn.Linear(cross_fusion_output_dim + pre_input_output_dim, final_fusion_output_dim)

        # ========================= Graph Readout Layer Configuration =========================
        # Linear transformation for single-value output
        self.readout = nn.Linear(final_fusion_output_dim, 1)
        # Build multi-layer perceptron with configurable architecture
        layers = []
        current_dim = num_entity
        # Add hidden layers
        for hidden_dim in linear_hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            # Add activation function using the helper function
            layers.append(create_linear_activation_layer(linear_activation))
            # Add dropout
            layers.append(nn.Dropout(linear_dropout_rate))
            current_dim = hidden_dim
        # Create the sequential model
        self.linear_repr = nn.Sequential(*layers)
        # =================================================================================

        # Final classification layer
        self.classifier = nn.Linear(linear_hidden_dims[-1], num_class)

        # Reset parameters for all learnable components
        self.reset_parameters()

    def reset_parameters(self):
        """Reset all learnable parameters in the model to their initial values."""
        # Reset graph encoder parameters
        self.encoder.reset_parameters()
        self.internal_encoder.reset_parameters()

        # Reset linear transformation layers
        self.name_linear_transform.reset_parameters()
        self.desc_linear_transform.reset_parameters()
        self.seq_linear_transform.reset_parameters()
        self.omic_linear_transform.reset_parameters()
        self.cross_modal_fusion.reset_parameters()
        self.pre_transform.reset_parameters()
        self.fusion.reset_parameters()

        # Reset readout layer
        if hasattr(self, 'readout'):
            self.readout.reset_parameters()
        
        # Reset linear representation layers
        if hasattr(self, 'linear_repr'):
            for layer in self.linear_repr:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        
        # Reset final classifier
        self.classifier.reset_parameters()

    def forward(self, x, pre_x, edge_index, internal_edge_index, ppi_edge_index,
                num_entity, x_name_emb, x_desc_emb, x_bio_emb, batch_size, index, analysis_output_dir):

        # Formalize the index_list by starting from index and range length of batch_size
        index_list = [index + i for i in range(batch_size)]
        # Formalize the num_node based on num_entity and batch_size
        num_node = num_entity * batch_size

        # =================== Multi-Modal Feature Integration ===================
        # Transform and expand embeddings for all batch samples
        name_features = self.name_linear_transform(x_name_emb)
        name_features = name_features.repeat(batch_size, 1)
        desc_features = self.desc_linear_transform(x_desc_emb)
        desc_features = desc_features.repeat(batch_size, 1)
        bio_features = self.seq_linear_transform(x_bio_emb)
        bio_features = bio_features.repeat(batch_size, 1)
        omic_features = self.omic_linear_transform(x)
        # Concatenate all modalities and fuse them
        multi_modal_features = torch.cat([name_features, desc_features, bio_features, omic_features], dim=-1)
        cross_modal_output = self.cross_modal_fusion(multi_modal_features)
        # ========================================================================

        # ================= Hierarchical Feature Fusion =======================
        # Combine pre-trained features with cross-modal features
        pre_features_transformed = self.pre_transform(pre_x)
        combined_features = torch.cat([pre_features_transformed, cross_modal_output], dim=-1)
        fused_features = self.fusion(combined_features)
        # ========================================================================

        # =================== Graph Neural Network Encoding ====================
        # Apply internal graph convolution with residual connection
        # import pdb; pdb.set_trace()
        internal_output = self.internal_encoder(fused_features, internal_edge_index)
        z = internal_output + x  # Residual connection
        # Apply main graph convolution
        z, attention_weights = self.encoder.forward_with_attention(z, ppi_edge_index)
        # Split attention weights by batch first, then by layers
        batch_attention_analysis = split_attention_by_batch_and_layer(attention_weights, batch_size, num_entity, index_list, remove_self_loops=True)
        # Save attention analysis if requested
        save_attention_analysis_to_folders(batch_attention_analysis, analysis_output_dir)
        # ========================================================================

        # =================== Graph-Level Linear Representation ==================
        # For linear readout - use MLP approach
        z = self.readout(z)  # Apply linear transformation: (B*N, D) -> (B*N, 1)
        z = z.view(batch_size, num_entity)  # Reshape to (B, N)
        z = self.linear_repr(z)  # Apply MLP: (B, N) -> (B, D')
        # ========================================================================

        # ===================== Final Classification ===========================
        output = self.classifier(z)
        _, pred = torch.max(output, dim=1)
        # ========================================================================

        return output, pred
    
    def loss(self, output, label):
        # import pdb; pdb.set_trace()
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
