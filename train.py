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
from CellTOSG_Foundation.utils import set_seed, tab_printer, get_dataset
from CellTOSG_Foundation.model import CellTOSG_Foundation, DegreeDecoder, EdgeDecoder, GNNEncoder
from CellTOSG_Foundation.mask import MaskEdge, MaskPath

# custom dataloader
from geo_loader.read_geograph import read_batch
from geo_loader.geograph_sampler import GeoGraphLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from enc_dec.geo_pretrain_gformer_decoder import GraphFormerDecoder

from CellTOSG_Foundation.lm_model import TextEncoder

from CellTOSG_Foundation.downstream import CellTOSG_Class


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


def build_model(args, device):
    text_encoder = TextEncoder(args.name_lm_model_path, device)

    internal_graph_encoder = GNNEncoder(args.num_omic_feature, args.input_dim, args.input_dim,
                            num_layers=args.internal_encoder_layers, dropout=args.encoder_dropout,
                            bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    graph_encoder = GNNEncoder(args.num_omic_feature, args.encoder_channels, args.hidden_channels,
                    num_layers=args.encoder_layers, dropout=args.encoder_dropout,
                    bn=args.bn, layer=args.layer, activation=args.encoder_activation)

    model = CellTOSG_Class(text_input_dim=args.text_emb_dim,
                    omic_input_dim=args.num_omic_feature,
                    input_dim=args.input_dim,
                    pre_input_dim=args.pre_input_dim,
                    output_dim=args.train_output_dim,
                    num_class=args.num_class,
                    text_encoder=text_encoder,
                    encoder=graph_encoder,
                    internal_encoder=internal_graph_encoder).to(device)
    return model

def train_model(train_dataset_loader, current_cell_num, num_entity, name_embeddings, desc_embeddings, seq_embeddings, pretrain_model, model, device, args, learning_rate):
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=learning_rate, eps=1e-7, weight_decay=1e-20)
    batch_loss = 0
    for batch_idx, data in enumerate(train_dataset_loader):
        optimizer.zero_grad()
        x = Variable(data.x.float(), requires_grad=False).to(device)
        internal_edge_index = Variable(data.internal_edge_index, requires_grad=False).to(device)
        ppi_edge_index = Variable(data.edge_index, requires_grad=False).to(device)
        edge_index = Variable(data.all_edge_index, requires_grad=False).to(device)
        label = Variable(data.label, requires_grad=False).to(device)

        # import pdb; pdb.set_trace()
        # Use pretrained model to get the embedding
        z = pretrain_model.internal_encoder(x, internal_edge_index) + x
        pre_x = pretrain_model.encoder.get_embedding(z, ppi_edge_index, mode='last') # mode='cat'
        output, ypred = model(x, pre_x, edge_index, internal_edge_index, ppi_edge_index, num_entity, name_embeddings, desc_embeddings, seq_embeddings, current_cell_num)
        loss = model.loss(output, label)
        loss.backward()
        batch_loss += loss.item()
        print('Label: ', label)
        print('Prediction: ', ypred)
        batch_acc = accuracy_score(label.cpu().numpy(), ypred.cpu().numpy())
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        # # check pretrain model parameters
        # state_dict = pretrain_model.internal_encoder.state_dict()
        # print(state_dict['convs.1.lin.weight'])
        # print(model.embedding.weight.data)
    torch.cuda.empty_cache()
    return model, batch_loss, batch_acc, ypred


def test_model(test_dataset_loader, current_cell_num, num_entity, name_embeddings, desc_embeddings, seq_embeddings, pretrain_model, model, device, args):
    batch_loss = 0
    all_ypred = np.zeros((1, 1))
    for batch_idx, data in enumerate(test_dataset_loader):
        x = Variable(data.x.float(), requires_grad=False).to(device)
        internal_edge_index = Variable(data.internal_edge_index, requires_grad=False).to(device)
        ppi_edge_index = Variable(data.edge_index, requires_grad=False).to(device)
        edge_index = Variable(data.all_edge_index, requires_grad=False).to(device)
        label = Variable(data.label, requires_grad=False).to(device)
        # Use pretrained model to get the embedding
        z = pretrain_model.internal_encoder(x, internal_edge_index) + x
        pre_x = pretrain_model.encoder.get_embedding(z, ppi_edge_index, mode='last') # mode='cat'
        output, ypred = model(x, pre_x, edge_index, internal_edge_index, ppi_edge_index, num_entity, name_embeddings, desc_embeddings, seq_embeddings, current_cell_num)
        loss = model.loss(output, label)
        batch_loss += loss.item()
        print('Label: ', label)
        print('Prediction: ', ypred)
        batch_acc = accuracy_score(label.cpu().numpy(), ypred.cpu().numpy())
        all_ypred = np.vstack((all_ypred, ypred.cpu().numpy().reshape(-1, 1)))
        all_ypred = np.delete(all_ypred, 0, axis=0)
    return model, batch_loss, batch_acc, all_ypred


def train(args, pretrain_model, device):
    # Load data
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
    xAll = xAll.reshape(xAll.shape[0], xAll.shape[1], 1)

    # process the xTr
    xTr = xAll.copy()
    num_cell = xTr.shape[0]
    num_entity = xTr.shape[1]
    # process the yTr
    unique_labels = np.unique(yAll)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    yAll = np.vectorize(label_mapping.get)(yAll)
    yAll = yAll.reshape(num_cell, -1)
    yTr = yAll.copy()
    print(xTr.shape, yTr.shape)

    num_cell = xTr.shape[0]
    num_entity = xTr.shape[1]

    all_edge_index = torch.from_numpy(np.load('./CellTOSG/edge_index.npy')).long()
    internal_edge_index = torch.from_numpy(np.load('./CellTOSG/internal_edge_index.npy')).long()
    ppi_edge_index = torch.from_numpy(np.load('./CellTOSG/ppi_edge_index.npy')).long()

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

    # Train the model depends on the task
    model = build_model(args, device)
    epoch_num = args.num_train_epoch
    learning_rate = args.train_lr
    train_batch_size = args.train_batch_size

    epoch_loss_list = []
    epoch_acc_list = []
    test_loss_list = []
    test_acc_list = []
    max_test_acc = 0
    max_test_acc_id = 0

    # Clean result previous epoch_i_pred files
    folder_name = 'epoch_' + str(epoch_num)
    path = './' + args.dataset_name + '-result/' + args.model_name + '/%s' % (folder_name)
    unit = 1
    # Ensure the parent directories exist
    os.makedirs('./' + args.dataset_name + '-result/' + args.model_name, exist_ok=True)
    while os.path.exists(path):
        path = './' + args.dataset_name + '-result/' + args.model_name + '/%s_%d' % (folder_name, unit)
        unit += 1
    os.mkdir(path)

    for i in range(1, epoch_num + 1):
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        print('---------------------------EPOCH: ' + str(i) + ' ---------------------------')
        model.train()
        epoch_ypred = np.zeros((1, 1))
        upper_index = 0
        batch_loss_list = []
        for index in range(0, num_cell, train_batch_size):
            if (index + train_batch_size) < num_cell:
                upper_index = index + train_batch_size
            else:
                upper_index = num_cell
            geo_train_datalist = read_batch(index, upper_index, xTr, yTr, num_feature, num_entity, all_edge_index, internal_edge_index, ppi_edge_index)
            train_dataset_loader = GeoGraphLoader.load_graph(geo_train_datalist, args.train_batch_size, args.train_num_workers)
            current_cell_num = upper_index - index # current batch size
            model, batch_loss, batch_acc, batch_ypred = train_model(train_dataset_loader, current_cell_num, num_entity, name_embeddings, desc_embeddings, seq_embeddings, pretrain_model, model, device, args, learning_rate)
            print('BATCH LOSS: ', batch_loss)
            print('BATCH ACCURACY: ', batch_acc)
            batch_loss_list.append(batch_loss)
            # PRESERVE PREDICTION OF BATCH TRAINING DATA
            batch_ypred = (Variable(batch_ypred).data).cpu().numpy().reshape(-1, 1)
            epoch_ypred = np.vstack((epoch_ypred, batch_ypred))
        epoch_loss = np.mean(batch_loss_list)
        print('TRAIN EPOCH ' + str(i) + ' LOSS: ', epoch_loss)
        epoch_loss_list.append(epoch_loss)
        epoch_ypred = np.delete(epoch_ypred, 0, axis = 0)
        # print('ITERATION NUMBER UNTIL NOW: ' + str(iteration_num))
        # Preserve acc corr for every epoch
        score_lists = list(yTr)
        score_list = [item for elem in score_lists for item in elem]
        epoch_ypred_lists = list(epoch_ypred)
        epoch_ypred_list = [item for elem in epoch_ypred_lists for item in elem]
        train_dict = {'label': score_list, 'prediction': epoch_ypred_list}
        tmp_training_input_df = pd.DataFrame(train_dict)
        # Calculating metrics
        accuracy = accuracy_score(tmp_training_input_df['label'], tmp_training_input_df['prediction'])
        tmp_training_input_df.to_csv(path + '/TrainingPred_' + str(i) + '.txt', index=False, header=True)
        epoch_acc_list.append(accuracy)

        conf_matrix = confusion_matrix(tmp_training_input_df['label'], tmp_training_input_df['prediction'])
        print('EPOCH ' + str(i) + ' TRAINING ACCURACY: ', accuracy)
        print('EPOCH ' + str(i) + ' TRAINING CONFUSION MATRIX: ', conf_matrix)

        print('\n-------------EPOCH TRAINING ACCURACY LIST: -------------')
        print(epoch_acc_list)
        print('\n-------------EPOCH TRAINING LOSS LIST: -------------')
        print(epoch_loss_list)

        # # # Test model on test dataset
        test_acc, test_loss, tmp_test_input_df = test(args, pretrain_model, model, device, i)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        tmp_test_input_df.to_csv(path + '/TestPred' + str(i) + '.txt', index=False, header=True)
        print('\n-------------EPOCH TEST ACCURACY LIST: -------------')
        print(test_acc_list)
        print('\n-------------EPOCH TEST MSE LOSS LIST: -------------')
        print(test_loss_list)
        # SAVE BEST TEST MODEL
        if test_acc >= max_test_acc:
            max_test_acc = test_acc
            max_test_acc_id = i
            # torch.save(model.state_dict(), path + '/best_train_model'+ str(i) +'.pt')
            torch.save(model.state_dict(), path + '/best_train_model.pt')
            tmp_training_input_df.to_csv(path + '/BestTrainingPred.txt', index=False, header=True)
            tmp_test_input_df.to_csv(path + '/BestTestPred.txt', index=False, header=True)
        print('\n-------------BEST TEST ACCURACY MODEL ID INFO:' + str(max_test_acc_id) + '-------------')
        print('--- TRAIN ---')
        print('BEST MODEL TRAIN LOSS: ', epoch_loss_list[max_test_acc_id - 1])
        print('BEST MODEL TRAIN ACCURACY: ', epoch_acc_list[max_test_acc_id - 1])
        print('--- TEST ---')
        print('BEST MODEL TEST LOSS: ', test_loss_list[max_test_acc_id - 1])
        print('BEST MODEL TEST ACCURACY: ', test_acc_list[max_test_acc_id - 1])


def test(args, pretrain_model, model, device, i):
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    # Load data
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    # Read the static data
    s_name_df = pd.read_csv('./CellTOSG/s_name.csv')
    s_desc_df = pd.read_csv('./CellTOSG/s_desc.csv')

    # Read these feature label files
    print('--- LOADING TRAINING FILES ... ---')
    x_file_path = './CellTOSG/brain_sc_output/processed_data/brain/alzheimer\'s_disease/alzheimer\'s_disease_X_partition_1.npy'
    y_file_path = './CellTOSG/brain_sc_output/processed_data/brain/alzheimer\'s_disease/alzheimer\'s_disease_Y_partition_1.npy'

    xAll = np.load(x_file_path)
    yAll = np.load(y_file_path)
    xAll = xAll.reshape(xAll.shape[0], xAll.shape[1], 1)

    # process the xTe
    xTe = xAll.copy()
    num_cell = xTe.shape[0]
    num_entity = xTe.shape[1]
    # process the yTe
    unique_labels = np.unique(yAll)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    yAll = np.vectorize(label_mapping.get)(yAll)
    yAll = yAll.reshape(num_cell, -1)
    yTe = yAll.copy()
    print(xTe.shape, yTe.shape)


    all_edge_index = torch.from_numpy(np.load('./CellTOSG/edge_index.npy')).long()
    internal_edge_index = torch.from_numpy(np.load('./CellTOSG/internal_edge_index.npy')).long()
    ppi_edge_index = torch.from_numpy(np.load('./CellTOSG/ppi_edge_index.npy')).long()

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

    test_batch_size = args.train_batch_size
    
    # Run test model
    model.eval()
    all_ypred = np.zeros((1, 1))
    upper_index = 0
    batch_loss_list = []
    for index in range(0, num_cell, test_batch_size):
        if (index + test_batch_size) < num_cell:
            upper_index = index + test_batch_size
        else:
            upper_index = num_cell
        geo_datalist = read_batch(index, upper_index, xTe, yTe, num_feature, num_entity, all_edge_index, internal_edge_index, ppi_edge_index)
        test_dataset_loader = GeoGraphLoader.load_graph(geo_datalist, args.train_batch_size, args.train_num_workers)
        print('TEST MODEL...')
        current_cell_num = upper_index - index # current batch size
        model, batch_loss, batch_acc, batch_ypred = test_model(test_dataset_loader, current_cell_num, num_entity, name_embeddings, desc_embeddings, seq_embeddings, pretrain_model, model, device, args)
        print('BATCH LOSS: ', batch_loss)
        batch_loss_list.append(batch_loss)
        print('BATCH ACCURACY: ', batch_acc)
        # PRESERVE PREDICTION OF BATCH TEST DATA
        batch_ypred = batch_ypred.reshape(-1, 1)
        all_ypred = np.vstack((all_ypred, batch_ypred))
    test_loss = np.mean(batch_loss_list)
    print('EPOCH ' + str(i) + ' TEST LOSS: ', test_loss)
    # Preserve accuracy for every epoch
    all_ypred = np.delete(all_ypred, 0, axis=0)
    all_ypred_lists = list(all_ypred)
    all_ypred_list = [item for elem in all_ypred_lists for item in elem]
    score_lists = list(yTe)
    score_list = [item for elem in score_lists for item in elem]
    test_dict = {'label': score_list, 'prediction': all_ypred_list}
    # import pdb; pdb.set_trace()
    tmp_test_input_df = pd.DataFrame(test_dict)
    # Calculating metrics
    accuracy = accuracy_score(tmp_test_input_df['label'], tmp_test_input_df['prediction'])
    conf_matrix = confusion_matrix(tmp_test_input_df['label'], tmp_test_input_df['prediction'])
    print('EPOCH ' + str(i) + ' TEST ACCURACY: ', accuracy)
    print('EPOCH ' + str(i) + ' TEST CONFUSION MATRIX: ', conf_matrix)
    test_acc = accuracy
    return test_acc, test_loss, tmp_test_input_df


def arg_parse():
    parser = argparse.ArgumentParser()
    # pre-training parameters
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for model and dataset. (default: 2025)')
    parser.add_argument('--pretrain', type=int, default=1, help='Whether to pretrain the model. (default: False)')
    parser.add_argument('--pretrain_text_batch_size', type=int, default=64, help='Batch size for pretraining text. (default: 64)')
    parser.add_argument('--name_lm_model_path', nargs='?', default='microsoft/deberta-v3-small', help='Path to the pretrained language model. (default: microsoft/deberta-v3-small)')

    parser.add_argument('--layer', nargs='?', default='gin', help='GNN layer, (default: gin)')
    parser.add_argument('--encoder_activation', nargs='?', default='elu', help='Activation function for GNN encoder, (default: elu)')

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
    parser.add_argument('--num_workers', dest = 'num_workers', type = int, default=0, help = 'Number of workers to load data.')

    parser.add_argument('--start', nargs='?', default='node', help='Which Type to sample starting nodes for random walks, (default: node)')
    parser.add_argument('--p', type=float, default=0.000001, help='Mask ratio or sample ratio for MaskEdge')

    parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
    parser.add_argument('--l2_normalize', action='store_true', help='Whether to use l2 normalize output embedding. (default: False)')
    parser.add_argument('--graphclas_weight_decay', type=float, default=1e-3, help='weight_decay for node classification training. (default: 1e-3)')

    parser.add_argument('--save_path', nargs='?', default='./Checkpoints/pretrained_models/pretrained_celltosg_foundation.pt', help='save path for model. (default: pretrained_celltosg_foundation.pt)')
    parser.add_argument('--device', type=int, default=0)

    # downstream task parameters
    parser.add_argument('--task', nargs='?', default='class', help='Task for training downstream tasks. (default: class)')
    parser.add_argument('--num_class', type=int, default=8, help='Number of classes for classification. (default: 2)')

    parser.add_argument('--train_lr', type=float, default=0.005, help='Learning rate for training. (default: 0.005)')
    parser.add_argument('--num_train_epoch', type=int, default=100, help='Number of training epochs. (default: 100)')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Batch size for training. (default: 4)')
    parser.add_argument('--train_num_workers', type=int, default=0, help='Number of workers to load data.')
    parser.add_argument('--fold_n', type=int, default=1, help='Fold number for training. (default: 1)')

    parser.add_argument('--pre_input_dim', type=int, default=8, help='Input feature dimension for pretraining. (default: 8)')
    parser.add_argument('--train_input_dim', type=int, default=1, help='Input feature dimension for training. (default: 1)')
    parser.add_argument('--train_hidden_dim', type=int, default=8, help='Hidden feature dimension for training. (default: 8)')
    parser.add_argument('--train_output_dim', type=int, default=8, help='Output feature dimension for training. (default: 8)')

    parser.add_argument('--dataset_name', nargs='?', default='AD', help='Datasets. (default: AD)')
    parser.add_argument('--model_name', nargs='?', default='CellTOSG-Class', help='Path to save training results. (default: train_result)')


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

    torch.cuda.empty_cache()
    # Load pretrain model
    pretrain_model = build_pretrain_model(args, device)
    pretrain_model.load_state_dict(torch.load(args.save_path))
    pretrain_model.eval()

    # Train the model
    train(args, pretrain_model, device)