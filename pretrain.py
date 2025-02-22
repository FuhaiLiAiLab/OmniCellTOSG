import os
import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm.auto import tqdm

from torch.autograd import Variable
from torch.utils.data import DataLoader

# custom modules
from maskgae.utils import set_seed, tab_printer, get_dataset
from maskgae.model import MaskGAE, DegreeDecoder, EdgeDecoder, GNNEncoder
from maskgae.mask import MaskEdge, MaskPath

# custom dataloader
from geo_loader.read_geograph import read_batch
from geo_loader.geograph_sampler import GeoGraphLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from enc_dec.geo_pretrain_gformer_decoder import GraphFormerDecoder

from maskgae.lm_model import TextEncoder


def build_pretrain_model(args, num_entity, device):
    mask = MaskEdge(p=args.p)

    text_encoder = TextEncoder(args.name_lm_model_path, device)

    graph_encoder = GNNEncoder(args.num_omic_feature, args.encoder_channels, args.hidden_channels,
                        num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                        bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    internal_graph_encoder = GNNEncoder(args.num_omic_feature, args.input_dim, args.input_dim,
                            num_layers=args.internal_encoder_layers, dropout=args.encoder_dropout,
                            bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    edge_decoder = EdgeDecoder(args.hidden_channels, args.decoder_channels,
                            num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    degree_decoder = DegreeDecoder(args.hidden_channels, args.decoder_channels,
                                num_layers=args.decoder_layers, dropout=args.decoder_dropout)

    pretrain_model = MaskGAE(input_dim=args.input_dim, 
                    num_node=num_entity,
                    text_encoder=text_encoder,
                    encoder=graph_encoder,
                    internal_encoder=internal_graph_encoder,
                    edge_decoder=edge_decoder,
                    degree_decoder=degree_decoder,
                    mask=mask).to(device)
    
    return pretrain_model


def pretrain_linkpred(pretrain_model, splits, args, device='cpu'):
    print('Start Training (Link Prediction Pretext Training)...')
    optimizer = torch.optim.Adam(pretrain_model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    batch_size = args.batch_size
    
    train_data = splits['train'].to(device)
    test_data = splits['test'].to(device)
    
    pretrain_model.reset_parameters()

    loss = pretrain_model.train_step(train_data, optimizer,
                                alpha=args.alpha, 
                                batch_size=args.batch_size)
    torch.save(pretrain_model.state_dict(), args.save_path)

    test_ap = pretrain_model.test_step(test_data, test_data.pos_edge_label_index, batch_size=batch_size)
    
    print(f'Link Prediction Pretraining Results:\n', f'AP: {test_ap:.2%}')
    return test_ap


def pretrain_foundation(args, device):
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

    # Build Pretrain Model
    pretrain_model = build_pretrain_model(args, num_entity, device)
    num_feature = args.num_omic_feature

    if args.pretrain==1:
        upper_index = 0
        batch_size = args.pretrain_batch_size

        for index in range(0, num_cell, batch_size):
            if (index + batch_size) < num_cell:
                upper_index = index + batch_size
            else:
                upper_index = num_cell
            geo_datalist = read_batch(index, upper_index, xAll, yAll, num_feature, num_entity, all_edge_index, internal_edge_index, ppi_edge_index)
            dataset_loader = GeoGraphLoader.load_graph(geo_datalist, args) # read by batch size

            for batch_idx, data in enumerate(dataset_loader):
                train_data, val_data, test_data = T.RandomLinkSplit(num_test=0.1, num_val=0.0,
                                                                is_undirected=False,
                                                                split_labels=True,
                                                                add_negative_train_samples=False)(data)
                if args.full_data:
                # Use full graph for pretraining
                    splits = dict(train=data, test=test_data)
                else:
                    splits = dict(train=train_data, test=test_data)
                print(f'Starting {index} - {upper_index}')
                pretrain_linkpred(pretrain_model, splits, args, device=device)
                print(f'Pretraining {upper_index} done!')
    else:
        pretrain_model.load_state_dict(torch.load(args.save_path))
    return pretrain_model


def arg_parse():
    parser = argparse.ArgumentParser()
    # pre-training parameters
    parser.add_argument('--dataset', nargs='?', default='UCSC', help='Datasets. (default: UCSC)')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for model and dataset. (default: 2025)')
    parser.add_argument('--pretrain', type=int, default=1, help='Whether to pretrain the model. (default: False)')
    parser.add_argument('--pretrain_batch_size', type=int, default=1, help='Batch size for pretraining. (default: 1)')

    parser.add_argument('--layer', nargs='?', default='gcn', help='GNN layer, (default: gcn)')
    parser.add_argument('--encoder_activation', nargs='?', default='elu', help='Activation function for GNN encoder, (default: elu)')

    parser.add_argument('--name_lm_model_path', nargs='?', default='microsoft/deberta-v3-small', help='Path to the pretrained language model. (default: microsoft/deberta-v3-small)')

    parser.add_argument('--num_omic_feature', type=int, default=1, help='Omic feature size. (default: 1)')

    parser.add_argument('--input_dim', type=int, default=1, help='Input feature dimension. (default: 1)')
    parser.add_argument('--encoder_channels', type=int, default=128, help='Channels of GNN encoder layers. (default: 1)')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Channels of hidden representation. (default: 1)')
    parser.add_argument('--decoder_channels', type=int, default=32, help='Channels of decoder layers. (default: 1)')

    parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 2)')
    parser.add_argument('--internal_encoder_layers', type=int, default=1, help='Number of layers for internal encoder. (default: 1)')
    parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
    parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.8)')
    parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')
    parser.add_argument('--alpha', type=float, default=0., help='loss weight for degree prediction. (default: 0.)')

    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for pre-training. (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for link prediction training. (default: 5e-5)')
    parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of batch size for link prediction training. (default: 32)')
    parser.add_argument('--num_workers', dest = 'num_workers', type = int, default=0, help = 'Number of workers to load data.')

    parser.add_argument('--start', nargs='?', default='node', help='Which Type to sample starting nodes for random walks, (default: node)')
    parser.add_argument('--p', type=float, default=0.0001, help='Mask ratio or sample ratio for MaskEdge')

    parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
    parser.add_argument('--l2_normalize', action='store_true', help='Whether to use l2 normalize output embedding. (default: False)')
    parser.add_argument('--graphclas_weight_decay', type=float, default=1e-3, help='weight_decay for node classification training. (default: 1e-3)')

    parser.add_argument('--epochs', type=int, default=5, help='Number of pre-training epochs. (default: 5)')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs. (default: 10)')
    parser.add_argument('--eval_period', type=int, default=30, help='(default: 30)')
    parser.add_argument('--save_path', nargs='?', default='pretrained_glm.pt', help='save path for model. (default: pretrained_glm.pt)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--full_data', action='store_true', help='Whether to use full data for pretraining. (default: False)')

    return parser.parse_args()


if __name__ == "__main__":
    # Set arguments and print
    args = arg_parse()
    print(tab_printer(args))
    # Check device
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if args.device < 0:
        device = 'cpu'
    else:
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    # Pretrain model
    pretrain_model = pretrain_foundation(args, device)
    