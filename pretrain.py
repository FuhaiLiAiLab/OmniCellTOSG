import os
import argparse
import pandas as pd
import numpy as np

import torch
import torch_geometric.transforms as T

# custom modules
from CellTOSG_Foundation.utils import tab_printer
from CellTOSG_Foundation.model import CellTOSG_Foundation, DegreeDecoder, EdgeDecoder, GNNEncoder
from CellTOSG_Foundation.mask import MaskEdge

# custom dataloader
from geo_loader.read_geograph import read_batch
from geo_loader.geograph_sampler import GeoGraphLoader

from CellTOSG_Foundation.lm_model import TextEncoder


def build_pretrain_model(args, device):
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

    pretrain_model = CellTOSG_Foundation(text_input_dim=args.text_emb_dim,
                    omic_input_dim=args.num_omic_feature,
                    input_dim=args.input_dim, 
                    text_encoder=text_encoder,
                    encoder=graph_encoder,
                    internal_encoder=internal_graph_encoder,
                    edge_decoder=edge_decoder,
                    degree_decoder=degree_decoder,
                    mask=mask).to(device)
    
    return pretrain_model



def pretrain_foundation(args, device):
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    # Read the static data
    s_name_df = pd.read_csv('./CellTOSG/s_name.csv')
    s_desc_df = pd.read_csv('./CellTOSG/s_desc.csv')

    # Read these feature label files
    print('--- LOADING TRAINING FILES ... ---')
    x_file_path = './CellTOSG/brain_sc_output/processed_data/brain/alzheimer\'s_disease/alzheimer\'s_disease_X_partition_0.npy'
    y_file_path = './CellTOSG/brain_sc_output/processed_data/brain/alzheimer\'s_disease/alzheimer\'s_disease_Y_partition_0.npy'

    xAll = np.load(x_file_path)
    yAll = np.load(y_file_path)
    print(xAll.shape, yAll.shape)

    num_cell = xAll.shape[0]
    yAll = yAll.reshape(num_cell, -1)

    all_edge_index = torch.from_numpy(np.load('./CellTOSG/edge_index.npy')).long()
    internal_edge_index = torch.from_numpy(np.load('./CellTOSG/internal_edge_index.npy')).long()
    ppi_edge_index = torch.from_numpy(np.load('./CellTOSG/ppi_edge_index.npy')).long()

    num_entity = xAll.shape[1]

    # Build Pretrain Model
    pretrain_model = build_pretrain_model(args, device)
    num_feature = args.num_omic_feature

    # # Use language model to embed the name and description
    # name_sentence_list = s_name_df['Name'].tolist()
    # desc_sentence_list = s_desc_df['Description'].tolist()
    # text_encoder = pretrain_model.text_encoder
    # text_encoder.load_model()
    # name_embeddings = text_encoder.generate_embeddings(name_sentence_list, batch_size=args.pretrain_text_batch_size, text_emb_dim=1)
    # desc_embeddings = text_encoder.generate_embeddings(desc_sentence_list, batch_size=args.pretrain_text_batch_size, text_emb_dim=1)

    name_embeddings = np.load('./CellTOSG/x_name_emb.npy').reshape(-1, args.text_emb_dim)
    desc_embeddings = np.load('./CellTOSG/x_desc_emb.npy').reshape(-1, args.text_emb_dim)
    seq_embeddings = np.load('./CellTOSG/x_bio_emb.npy').reshape(-1, args.text_emb_dim)

    # load textual embeddings into torch tensor
    name_embeddings = torch.from_numpy(name_embeddings).float().to(device)
    desc_embeddings = torch.from_numpy(desc_embeddings).float().to(device)
    seq_embeddings = torch.from_numpy(seq_embeddings).float().to(device)

    if args.pretrain==1:
        upper_index = 0
        batch_size = args.pretrain_batch_size

        batch_avg_loss_list = []
        all_step_avg_loss_list = []
        batch_auc_list = []
        batch_acc_list = []
        best_loss = 1000
        for index in range(0, num_cell, batch_size):
            if (index + batch_size) < num_cell:
                upper_index = index + batch_size
            else:
                upper_index = num_cell
            current_cell_num = upper_index - index
            geo_datalist = read_batch(index, upper_index, xAll, yAll, num_feature, num_entity, all_edge_index, internal_edge_index, ppi_edge_index)
            dataset_loader = GeoGraphLoader.load_graph(geo_datalist, args.pretrain_batch_size, args.pretrain_num_workers) # read by batch size

            for batch_idx, data in enumerate(dataset_loader):
                print(f'Starting {index} - {upper_index}')
                print('Start Training (Link Prediction Pretext Training)...')
                optimizer = torch.optim.Adam(pretrain_model.parameters(),
                                            lr=args.lr,
                                            weight_decay=args.weight_decay)

                # import pdb; pdb.set_trace()

                train_data, val_data, test_data = T.RandomLinkSplit(num_test=0.1, num_val=0.0,
                                                                is_undirected=False,
                                                                split_labels=True,
                                                                add_negative_train_samples=False)(data)
                
                pretrain_model.reset_parameters()
                train_data = train_data.to(device)
                avg_loss, step_avg_loss_list = pretrain_model.train_step(train_data,
                                            num_entity,
                                            name_embeddings, desc_embeddings, seq_embeddings,
                                            optimizer,
                                            alpha=args.alpha, 
                                            batch_size=current_cell_num)
                batch_avg_loss_list.append(avg_loss)
                all_step_avg_loss_list.extend(step_avg_loss_list)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(pretrain_model.state_dict(), args.save_path)
                # save loss list to text file
                with open(args.save_path.replace('.pt', '_batch_avg_loss_list.txt'), 'w') as f:
                    for item in batch_avg_loss_list:
                        f.write("%s\n" % item)
                with open(args.save_path.replace('.pt', '_all_step_avg_loss_list.txt'), 'w') as f:
                    for item in all_step_avg_loss_list:
                        f.write("%s\n" % item)

                test_data = test_data.to(device)
                test_auc, test_ap = pretrain_model.test_step(test_data, 
                                        test_data.pos_edge_label_index, 
                                        test_data.neg_edge_label_index) 
                batch_auc_list.append(test_auc)
                batch_acc_list.append(test_ap)
                # save auc list to text file
                with open(args.save_path.replace('.pt', '_batch_auc.txt'), 'w') as f:
                    for item in batch_auc_list:
                        f.write("%s\n" % item) 
                with open(args.save_path.replace('.pt', '_batch_ap.txt'), 'w') as f:
                    for item in batch_acc_list:
                        f.write("%s\n" % item) 
                print(f'Link Prediction Pretraining Results:\n'
                    f'AUC: {test_auc:.2%}',
                    f'AP: {test_ap:.2%}')
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
    parser.add_argument('--pretrain_batch_size', type=int, default=2, help='Batch size for pretraining. (default: 2)')
    parser.add_argument('--pretrain_text_batch_size', type=int, default=64, help='Batch size for pretraining text. (default: 64)')
    parser.add_argument('--name_lm_model_path', nargs='?', default='microsoft/deberta-v3-small', help='Path to the pretrained language model. (default: microsoft/deberta-v3-small)')

    parser.add_argument('--layer', nargs='?', default='gcn', help='GNN layer, (default: gcn)')
    parser.add_argument('--encoder_activation', nargs='?', default='leaky_relu', help='Activation function for GNN encoder, (default: leaky_relu)')

    parser.add_argument('--num_omic_feature', type=int, default=1, help='Omic feature size. (default: 1)')
    parser.add_argument('--text_emb_dim', type=int, default=1, help='Text embedding dimension. (default: 1)')

    parser.add_argument('--input_dim', type=int, default=1, help='Input feature dimension. (default: 1)')
    parser.add_argument('--encoder_channels', type=int, default=8, help='Channels of GNN encoder layers. (default: 8)')
    parser.add_argument('--hidden_channels', type=int, default=8, help='Channels of hidden representation. (default: 8)')
    parser.add_argument('--decoder_channels', type=int, default=4, help='Channels of decoder layers. (default: 4)')

    parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 2)')
    parser.add_argument('--internal_encoder_layers', type=int, default=1, help='Number of layers for internal encoder. (default: 1)')
    parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
    parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.8)')
    parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')
    parser.add_argument('--alpha', type=float, default=0., help='loss weight for degree prediction. (default: 0.)')

    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for pre-training. (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay for link prediction training. (default: 5e-5)')
    parser.add_argument('--grad_norm', type=float, default=1.0, help='grad_norm for training. (default: 1.0.)')
    parser.add_argument('--pretrain_num_workers', dest = 'pretrain_num_workers', type = int, default=0, help = 'Number of workers to load data.')

    parser.add_argument('--start', nargs='?', default='node', help='Which Type to sample starting nodes for random walks, (default: node)')
    parser.add_argument('--p', type=float, default=0.00001, help='Mask ratio for MaskEdge')

    parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
    parser.add_argument('--l2_normalize', action='store_true', help='Whether to use l2 normalize output embedding. (default: False)')
    parser.add_argument('--graphclas_weight_decay', type=float, default=1e-3, help='weight_decay for node classification training. (default: 1e-3)')

    parser.add_argument('--save_path', nargs='?', default='./Checkpoints/pretrained_models/pretrained_celltosg_foundation.pt', help='save path for model. (default: pretrained_celltosg_foundation.pt)')
    parser.add_argument('--device', type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    # Set arguments and print
    args = arg_parse()
    print(tab_printer(args))
    # Check device
    if args.device < 0:
        device = 'cpu'
    else:
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    # Pretrain model
    pretrain_model = pretrain_foundation(args, device)
    