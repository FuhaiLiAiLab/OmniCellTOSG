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

# custom dataloader
from GeoDataLoader.read_geograph import read_batch
from GeoDataLoader.geograph_sampler import GeoGraphLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

from GraphModel.model import GraphDecoder
from GraphModel.utils import tab_printer


def build_model(args, num_entity, device):
    model = GraphDecoder(model_name=args.model_name,
                        input_dim=args.train_input_dim, 
                        hidden_dim=args.train_hidden_dim, 
                        embedding_dim=args.train_embedding_dim, 
                        num_node=num_entity, 
                        num_head=args.train_num_head, 
                        device=device, 
                        num_class=args.num_class).to(device)
    return model


def train_model(train_dataset_loader, current_cell_num, model, device, args, learning_rate):
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
        output, ypred = model(x, edge_index, internal_edge_index, ppi_edge_index, current_cell_num)
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


def test_model(test_dataset_loader, current_cell_num, model, device, args):
    batch_loss = 0
    all_ypred = np.zeros((1, 1))
    for batch_idx, data in enumerate(test_dataset_loader):
        x = Variable(data.x.float(), requires_grad=False).to(device)
        internal_edge_index = Variable(data.internal_edge_index, requires_grad=False).to(device)
        ppi_edge_index = Variable(data.edge_index, requires_grad=False).to(device)
        edge_index = Variable(data.all_edge_index, requires_grad=False).to(device)
        label = Variable(data.label, requires_grad=False).to(device)

        output, ypred = model(x, edge_index, internal_edge_index, ppi_edge_index, current_cell_num)
        loss = model.loss(output, label)
        batch_loss += loss.item()
        print('Label: ', label)
        print('Prediction: ', ypred)
        batch_acc = accuracy_score(label.cpu().numpy(), ypred.cpu().numpy())
        all_ypred = np.vstack((all_ypred, ypred.cpu().numpy().reshape(-1, 1)))
        all_ypred = np.delete(all_ypred, 0, axis=0)
    return model, batch_loss, batch_acc, all_ypred


def train(args, device):
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
    num_feature = args.num_omic_feature

    # Train the model depends on the task
    model = build_model(args, num_entity, device)
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
    path = './' + args.train_result_folder  + '/' + args.dataset_name + '/' + args.model_name + '/%s' % (folder_name)
    unit = 1
    # Ensure the parent directories exist
    os.makedirs('./' + args.train_result_folder  + '/' + args.dataset_name + '/' + args.model_name, exist_ok=True)
    while os.path.exists(path):
        path = './' + args.train_result_folder  + '/' + args.dataset_name + '/' + args.model_name + '/%s_%d' % (folder_name, unit)
        unit += 1
    os.mkdir(path)

    for i in range(1, epoch_num + 1):
        for _ in range(5):
            print('-------------------------- EPOCH ' + str(i) + ' START --------------------------')
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
            model, batch_loss, batch_acc, batch_ypred = train_model(train_dataset_loader, current_cell_num, model, device, args, learning_rate)
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
        test_acc, test_loss, tmp_test_input_df = test(args, model, device, i)
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


def test(args, model, device, i):
    for _ in range(5):
        print('-------------------------- TEST START --------------------------')

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
    num_feature = args.num_omic_feature
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
        model, batch_loss, batch_acc, batch_ypred = test_model(test_dataset_loader, current_cell_num, model, device, args)
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

    # dataset loading parameters
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for model and dataset. (default: 2025)')
    parser.add_argument('--train_text', type=bool, default=False, help='Whether to train text embeddings. (default: False)')
    parser.add_argument('--train_bio', type=bool, default=False, help='Whether to train bio-sequence embeddings. (default: False)')
    
    # Training arguments
    parser.add_argument('--device', type=int, default=0, help='Device to use for training (default: 0)')
    parser.add_argument('--num_train_epoch', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Training batch size (default: 4)')
    parser.add_argument('--train_lr', type=float, default=0.001, help='Learning rate for training (default: 0.001)')

    parser.add_argument('--num_omic_feature', type=int, default=1, help='Number of omic features (default: 1)')
    parser.add_argument('--num_class', type=int, default=8, help='Number of classes (default: 8)')
    parser.add_argument('--train_input_dim', type=int, default=1, help='Input dimension for training (default: 1)')
    parser.add_argument('--train_hidden_dim', type=int, default=8, help='Hidden dimension for training (default: 8)')
    parser.add_argument('--train_embedding_dim', type=int, default=8, help='Embedding dimension for training (default: 8)')
    parser.add_argument('--train_num_head', type=int, default=2, help='Number of heads in the model (default: 2)')
    parser.add_argument('--train_num_workers', type=int, default=0, help='Number of workers for data loading (default: 0)')
    
    parser.add_argument('--train_result_folder', nargs='?', default='Results', help='Path to save training results. (default: Results)')
    parser.add_argument('--dataset_name', nargs='?', default='AD', help='Datasets. (default: AD)')
    parser.add_argument('--model_name', nargs='?', default='gin', help='Model name. (default: gin)')

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
    # Train the model
    train(args, device)