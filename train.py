import os
import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

# custom dataloader
from GeoDataLoader.read_geograph import read_batch
from GeoDataLoader.geograph_sampler import GeoGraphLoader

# custom modules
from CellTOSG_Foundation.lm_model import TextEncoder, RNAGPT_LM, ProtGPT_LM
from CellTOSG_Foundation.downstream import CellTOSG_Class
from CellTOSG_Foundation.utils import tab_printer
from CellTOSG_Foundation.model import CellTOSG_Foundation, DegreeDecoder, EdgeDecoder, GNNEncoder
from CellTOSG_Foundation.mask import MaskEdge
from CellTOSG_Foundation.lm_model import TextEncoder, RNAGPT_LM, ProtGPT_LM
from CellTOSG_Loader import CellTOSGDataLoader


def build_pretrain_model(args, device):
    mask = MaskEdge(p=args.p)

    text_encoder = TextEncoder(args.text_lm_model_path, device)

    rna_seq_encoder = RNAGPT_LM(args.rna_seq_lm_model_path, args.rna_model_name, device)

    prot_seq_encoder = ProtGPT_LM(args.prot_model_name, device)

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

    pretrain_model = CellTOSG_Foundation(text_input_dim=args.lm_emb_dim,
                    omic_input_dim=args.num_omic_feature,
                    input_dim=args.input_dim, 
                    text_encoder=text_encoder,
                    rna_seq_encoder=rna_seq_encoder,
                    prot_seq_encoder=prot_seq_encoder,
                    encoder=graph_encoder,
                    internal_encoder=internal_graph_encoder,
                    edge_decoder=edge_decoder,
                    degree_decoder=degree_decoder,
                    mask=mask).to(device)
    
    return pretrain_model


def pre_embed_text(args, dataset, pretrain_model, device):
    """
    Prepare text and biological sequence embeddings.
    
    Args:
        args: Command line arguments
        dataset: Dataset object containing the data
        pretrain_model: Pretrained model with encoders
        device: Device to load tensors on
    
    Returns:
        tuple: (x_name_emb, x_desc_emb, x_bio_emb) as torch tensors
    """
    if args.train_text:
        s_name = dataset.s_name
        s_desc = dataset.s_desc
        # Use language model to embed the name and description
        name_sentence_list = s_name['Name'].tolist()
        name_sentence_list = [str(name) for name in name_sentence_list]
        desc_sentence_list = s_desc['Description'].tolist()
        desc_sentence_list = [str(desc) for desc in desc_sentence_list]
        text_encoder = pretrain_model.text_encoder
        text_encoder.load_model()
        x_name_emb = text_encoder.generate_embeddings(name_sentence_list, batch_size=args.pretrain_text_batch_size, text_emb_dim=args.lm_emb_dim)
        x_desc_emb = text_encoder.generate_embeddings(desc_sentence_list, batch_size=args.pretrain_text_batch_size, text_emb_dim=args.lm_emb_dim)
    else:
        # Use pre-computed embeddings
        x_name_emb = dataset.x_name_emb
        x_desc_emb = dataset.x_desc_emb

    if args.train_bio:
        s_bio = dataset.s_bio
        # Use language model to embed the RNA and protein sequences
        # sequence list where type == transcript
        rna_seq_list = s_bio[s_bio['Type'] == 'Transcript']['Sequence'].tolist()
        rna_seq_encoder = pretrain_model.rna_seq_encoder
        rna_seq_encoder.load_model()
        rna_replaced_seq_list = [' ' if type(i) == float else i.replace('U', 'T') for i in rna_seq_list]
        rna_seq_embeddings = rna_seq_encoder.generate_embeddings(rna_replaced_seq_list, batch_size=args.pretrain_text_batch_size, max_len=args.rna_seq_max_len, seq_emb_dim=args.lm_emb_dim)
        # sequence list where type == protein
        prot_seq_list = s_bio[s_bio['Type'] == 'Protein']['Sequence'].tolist()
        prot_seq_encoder = pretrain_model.prot_seq_encoder
        prot_seq_encoder.load_model()
        prot_replaced_seq_list = ['X' if type(i) == float else i for i in prot_seq_list]
        prot_seq_embeddings = prot_seq_encoder.generate_embeddings(prot_replaced_seq_list, seq_emb_dim=args.lm_emb_dim)
        x_bio_emb = np.concatenate((rna_seq_embeddings, prot_seq_embeddings), axis=0)
    else:
        # Use pre-computed embeddings
        x_bio_emb = dataset.x_bio_emb

    return x_name_emb, x_desc_emb, x_bio_emb


def split_dataset(xAll, yAll, args, output_dir):
    """
    Split dataset into train and test sets ensuring relatively even distribution for each class.
    Maps both numerical and textual values to vectorized values and saves mapping dictionary.
    
    Args:
        xAll: Input features tensor
        yAll: Labels tensor
        args: Command line arguments containing split parameters
        output_dir: Directory to save mapping dictionary
    
    Returns:
        tuple: (xTr, xTe, yTr, yTe, num_entity, num_feature)
    """    
    # Get unique values from yAll (works for both numerical and textual)
    unique_values = torch.unique(yAll)
    num_classes = len(unique_values)
    print("\n")
    print(f"Original unique values: {unique_values}")
    print(f"Total number of classes: {num_classes}")
    
    # Create mapping dictionary from original values to indices
    if yAll.dtype == torch.long or yAll.dtype == torch.int:
        # For numerical values
        value_to_index = {value.item(): index for index, value in enumerate(unique_values)}
    else:
        # For textual values (if stored as strings in tensor)
        value_to_index = {str(value): index for index, value in enumerate(unique_values)}
    print("Value to index mapping:", value_to_index)
    
    # Save mapping dictionary to CSV
    mapping_df = pd.DataFrame([
        {"original_value": k, "mapped_index": v} 
        for k, v in value_to_index.items()
    ])
    mapping_csv_path = os.path.join(output_dir, "label_mapping_dict.csv")
    mapping_df.to_csv(mapping_csv_path, index=False)
    print(f"Mapping dictionary saved to: {mapping_csv_path}")

    # Apply mapping to yAll
    yAll_mapped = torch.zeros_like(yAll, dtype=torch.long)
    for original_value, new_index in value_to_index.items():
        if yAll.dtype == torch.long or yAll.dtype == torch.int:
            mask = (yAll == original_value)
        else:
            mask = (yAll == str(original_value))
        yAll_mapped[mask] = new_index
    yAll = yAll_mapped
    yAll = yAll.reshape(-1, 1)
    
    # Print class distribution
    print("\nOriginal dataset class distribution:")
    for class_idx in range(num_classes):
        class_count = torch.sum(yAll == class_idx)
        percentage = (class_count / len(yAll)) * 100
        original_value = list(value_to_index.keys())[class_idx]
        print(f"Class {class_idx} ('{original_value}'): {class_count} samples ({percentage:.1f}%)")

    # Get dataset size and create indices
    dataset_size = xAll.shape[0]
    indices = torch.arange(dataset_size)

    # Split indices with stratification to ensure balanced distribution
    train_indices, test_indices = train_test_split(
        indices.numpy(), 
        test_size=1-args.train_test_split_ratio, 
        random_state=args.train_test_random_seed,
        stratify=yAll.cpu().numpy()  # Ensure balanced split across all classes
    )

    # Convert indices back to torch tensors
    train_indices = torch.from_numpy(train_indices).long()
    test_indices = torch.from_numpy(test_indices).long()

    # Use indices to split the tensors
    xTr = xAll[train_indices]
    xTe = xAll[test_indices]
    yTr = yAll[train_indices]
    yTe = yAll[test_indices]

    # Get dimensions
    train_num_cell = xTr.shape[0]
    test_num_cell = xTe.shape[0]
    num_entity = xTr.shape[1]
    num_feature = args.num_omic_feature
    print(f"\nDataset split summary:")
    print(f"Training samples: {train_num_cell}")
    print(f"Testing samples: {test_num_cell}")
    print(f"Number of entities: {num_entity}")
    print(f"Number of classes: {num_classes}")
    
    # Print class distribution in train and test sets
    print("\nTraining set class distribution:")
    for class_idx in range(num_classes):
        class_count = torch.sum(yTr == class_idx)
        percentage = (class_count / train_num_cell) * 100
        original_value = list(value_to_index.keys())[class_idx]
        print(f"Class {class_idx} ('{original_value}'): {class_count} samples ({percentage:.1f}%)")
    print("\nTesting set class distribution:")
    for class_idx in range(num_classes):
        class_count = torch.sum(yTe == class_idx)
        percentage = (class_count / test_num_cell) * 100
        original_value = list(value_to_index.keys())[class_idx]
        print(f"Class {class_idx} ('{original_value}'): {class_count} samples ({percentage:.1f}%)")

    return xTr, xTe, yTr, yTe, num_classes


def build_model(args, device):
    text_encoder = TextEncoder(args.text_lm_model_path, device)

    rna_seq_encoder = RNAGPT_LM(args.rna_seq_lm_model_path, args.rna_model_name, device)

    prot_seq_encoder = ProtGPT_LM(args.prot_model_name, device)

    internal_graph_encoder = GNNEncoder(args.num_omic_feature, args.input_dim, args.input_dim,
                            num_layers=args.train_internal_encoder_layers, dropout=args.encoder_dropout,
                            bn=args.bn, layer=args.layer, activation=args.train_encoder_activation)

    graph_encoder = GNNEncoder(args.num_omic_feature, args.encoder_channels, args.hidden_channels,
                    num_layers=args.train_encoder_layers, dropout=args.encoder_dropout,
                    bn=args.bn, layer=args.layer, activation=args.train_encoder_activation)

    model = CellTOSG_Class(text_input_dim=args.lm_emb_dim,
                    omic_input_dim=args.num_omic_feature,
                    input_dim=args.input_dim,
                    pre_input_dim=args.pre_input_dim,
                    output_dim=args.train_output_dim,
                    num_class=args.num_class,
                    text_encoder=text_encoder,
                    rna_seq_encoder=rna_seq_encoder,
                    prot_seq_encoder=prot_seq_encoder,
                    encoder=graph_encoder,
                    internal_encoder=internal_graph_encoder).to(device)
    return model


def write_best_model_info(path, max_test_acc_id, epoch_loss_list, epoch_acc_list, test_loss_list, test_acc_list):
    best_model_info = (
        f'\n-------------BEST TEST ACCURACY MODEL ID INFO: {max_test_acc_id} -------------\n'
        '--- TRAIN ---\n'
        f'BEST MODEL TRAIN LOSS: {epoch_loss_list[max_test_acc_id - 1]}\n'
        f'BEST MODEL TRAIN ACCURACY: {epoch_acc_list[max_test_acc_id - 1]}\n'
        '--- TEST ---\n'
        f'BEST MODEL TEST LOSS: {test_loss_list[max_test_acc_id - 1]}\n'
        f'BEST MODEL TEST ACCURACY: {test_acc_list[max_test_acc_id - 1]}\n'
    )
    with open(os.path.join(path, 'best_model_info.txt'), 'w') as file:
        file.write(best_model_info)


def train_model(train_dataset_loader, current_cell_num, num_entity, name_embeddings, desc_embeddings, seq_embeddings, pretrain_model, model, device, args):
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.train_lr, eps=args.train_eps, weight_decay=args.train_weight_decay)
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


def train(args, pretrain_model, model, device, xTr, xTe, yTr, yTe, all_edge_index, internal_edge_index, ppi_edge_index, x_name_emb, x_desc_emb, x_bio_emb):
    train_num_cell = xTr.shape[0]
    num_entity = xTr.shape[1]
    epoch_num = args.num_train_epoch
    train_batch_size = args.train_batch_size
    num_feature = args.num_omic_feature

    epoch_loss_list = []
    epoch_acc_list = []
    test_loss_list = []
    test_acc_list = []
    max_test_acc = 0
    max_test_acc_id = 0

    # Clean result previous epoch_i_pred files
    folder_name = 'epoch_' + str(epoch_num)
    path = './' + args.train_result_folder  + '/' + args.name + '/' + args.model_name + '/%s' % (folder_name)
    unit = 1
    # Ensure the parent directories exist
    os.makedirs('./' + args.train_result_folder  + '/' + args.name + '/' + args.model_name, exist_ok=True)
    while os.path.exists(path):
        path = './' + args.train_result_folder  + '/' + args.name + '/' + args.model_name + '/%s_%d' % (folder_name, unit)
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
        for index in range(0, train_num_cell, train_batch_size):
            if (index + train_batch_size) < train_num_cell:
                upper_index = index + train_batch_size
            else:
                upper_index = train_num_cell
            geo_train_datalist = read_batch(index, upper_index, xTr, yTr, num_feature, num_entity, all_edge_index, internal_edge_index, ppi_edge_index)
            train_dataset_loader = GeoGraphLoader.load_graph(geo_train_datalist, args.train_batch_size, args.train_num_workers)
            current_cell_num = upper_index - index # current batch size
            model, batch_loss, batch_acc, batch_ypred = train_model(train_dataset_loader, current_cell_num, num_entity, x_name_emb, x_desc_emb, x_bio_emb, pretrain_model, model, device, args)
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
        score_lists = yTr.cpu().numpy().tolist()
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
        test_acc, test_loss, tmp_test_input_df = test(args, pretrain_model, model, xTe, yTe, all_edge_index, internal_edge_index, ppi_edge_index, x_name_emb, x_desc_emb, x_bio_emb, device, i)
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
            write_best_model_info(path, max_test_acc_id, epoch_loss_list, epoch_acc_list, test_loss_list, test_acc_list)
        print('\n-------------BEST TEST ACCURACY MODEL ID INFO:' + str(max_test_acc_id) + '-------------')
        print('--- TRAIN ---')
        print('BEST MODEL TRAIN LOSS: ', epoch_loss_list[max_test_acc_id - 1])
        print('BEST MODEL TRAIN ACCURACY: ', epoch_acc_list[max_test_acc_id - 1])
        print('--- TEST ---')
        print('BEST MODEL TEST LOSS: ', test_loss_list[max_test_acc_id - 1])
        print('BEST MODEL TEST ACCURACY: ', test_acc_list[max_test_acc_id - 1])


def test(args, pretrain_model, model, xTe, yTe, all_edge_index, internal_edge_index, ppi_edge_index, x_name_emb, x_desc_emb, x_bio_emb, device, i):
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    print('-------------------------- TEST START --------------------------')
    
    print('--- LOADING TESTING FILES ... ---')
    print('xTe: ', xTe.shape)
    print('yTe: ', yTe.shape)
    test_num_cell = xTe.shape[0]
    num_entity = xTe.shape[1]
    num_feature = args.num_omic_feature
    test_batch_size = args.train_batch_size
    
    # Run test model
    model.eval()
    all_ypred = np.zeros((1, 1))
    upper_index = 0
    batch_loss_list = []
    for index in range(0, test_num_cell, test_batch_size):
        if (index + test_batch_size) < test_num_cell:
            upper_index = index + test_batch_size
        else:
            upper_index = test_num_cell
        geo_datalist = read_batch(index, upper_index, xTe, yTe, num_feature, num_entity, all_edge_index, internal_edge_index, ppi_edge_index)
        test_dataset_loader = GeoGraphLoader.load_graph(geo_datalist, args.train_batch_size, args.train_num_workers)
        print('TEST MODEL...')
        current_cell_num = upper_index - index # current batch size
        model, batch_loss, batch_acc, batch_ypred = test_model(test_dataset_loader, current_cell_num, num_entity, x_name_emb, x_desc_emb, x_bio_emb, pretrain_model, model, device, args)
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
    score_lists = yTe.cpu().numpy().tolist()
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

    #####################################################################################################################################################################
    # dataset loading parameters
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for model and dataset. (default: 2025)')
    parser.add_argument('--data_root', type=str, default='/storage1/fs1/fuhai.li/Active/tianqi.x/OmniCellTOSG/dataset_outputs', help='Root directory for dataset.')
    parser.add_argument('--categories', type=str, default='get_organ_disease', help='Categories for dataset. (default: get_organ_disease)')
    parser.add_argument('--tissue_general', type=str, default='brain', help='General tissue type for dataset. (default: brain)')
    parser.add_argument('--tissue', type=str, default='Cerebral Cortex', help='Specific tissue type for dataset. (default: Cerebral Cortex)')
    parser.add_argument('--suspension_type', type=str, default='nucleus', help='Suspension type for dataset. (default: nucleus)')
    parser.add_argument('--cell_type', type=str, default='neuronal', help='Cell type for dataset. (default: neuronal)')
    parser.add_argument('--disease_name', type=str, default="Alzheimer's Disease", help='Disease name for dataset.')
    parser.add_argument('--gender', type=str, default='female', help='Gender for dataset. (default: female)')
    parser.add_argument('--downstream_task', type=str, default='disease', help='Downstream task for dataset. (default: disease)')  # One of {"disease", "gender", "cell_type"}.
    parser.add_argument('--label_column', type=str, default='disease', help='Label column for dataset. (default: disease)')  # One of {"disease", "gender", "cell_type"}.

    parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle dataset. (default: True)')
    parser.add_argument('--sample_ratio', type=float, default=0.01, help='Sample ratio for dataset. (default: 0.01)')
    parser.add_argument('--sample_size', type=int, default=None, help='Sample size for dataset. (default: None)')
    parser.add_argument('--balanced', type=bool, default=True, help='Whether to balance dataset. (default: True)')
    parser.add_argument('--random_state', type=int, default=2025, help='Random state for dataset. (default: 2025)')
    parser.add_argument('--train_text', type=bool, default=False, help='Whether to train text embeddings. (default: False)')
    parser.add_argument('--train_bio', type=bool, default=False, help='Whether to train bio-sequence embeddings. (default: False)')
    parser.add_argument('--dataset_output_dir', type=str, default='./Output/data_ad_disease', help='Directory to save dataset outputs. (default: ./Output/data_ad_disease)')
    
    #####################################################################################################################################################################
    # pre-training parameters
    parser.add_argument('--pretrain_text_batch_size', type=int, default=64, help='Batch size for pretraining text. (default: 64)')
    parser.add_argument('--text_lm_model_path', type=str, default='microsoft/deberta-v3-small', help='Path to the pretrained language model. (default: microsoft/deberta-v3-small)')
    parser.add_argument('--rna_seq_lm_model_path', type=str, default='./Checkpoints/pretrained_dnagpt', help='Path to the pretrained RNA language model. (default: ./Checkpoints/pretrained_dnagpt)')
    parser.add_argument('--rna_model_name', type=str, default='dna_gpt0.1b_h', help='Name of the pretrained RNA language model. (default: dna_gpt0.1b_h)')
    parser.add_argument('--prot_model_name', type=str, default='nferruz/ProtGPT2', help='Name of the pretrained protein language model. (default: nferruz/ProtGPT2)')

    parser.add_argument('--layer', type=str, default='gcn', help='GNN layer type. (default: gcn)')
    parser.add_argument('--encoder_activation', type=str, default='leaky_relu', help='Activation function for GNN encoder. (default: leaky_relu)')

    parser.add_argument('--num_omic_feature', type=int, default=1, help='Omic feature size. (default: 1)')
    parser.add_argument('--lm_emb_dim', type=int, default=1, help='Text embedding dimension. (default: 1)')

    parser.add_argument('--input_dim', type=int, default=1, help='Input feature dimension. (default: 1)')
    parser.add_argument('--encoder_channels', type=int, default=8, help='Channels of GNN encoder layers. (default: 8)')
    parser.add_argument('--hidden_channels', type=int, default=8, help='Channels of hidden representation. (default: 8)')
    parser.add_argument('--decoder_channels', type=int, default=4, help='Channels of decoder layers. (default: 4)')

    parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers for encoder. (default: 2)')
    parser.add_argument('--internal_encoder_layers', type=int, default=1, help='Number of layers for internal encoder. (default: 1)')
    parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders. (default: 2)')
    parser.add_argument('--encoder_dropout', type=float, default=0.8, help='Dropout probability of encoder. (default: 0.8)')
    parser.add_argument('--decoder_dropout', type=float, default=0.2, help='Dropout probability of decoder. (default: 0.2)')

    parser.add_argument('--pretrained_model_path', type=str, default='./Checkpoints/pretrained_models/pretrained_celltosg_foundation.pt', help='Pretrained model path. (default: ./Checkpoints/pretrained_models/pretrained_celltosg_foundation.pt)')
    parser.add_argument('--device', type=int, default=0, help='Device ID for GPU. (default: 0)')

    ###############################################################################################################################################################################
    # downstream task parameters
    parser.add_argument('--train_test_split_ratio', type=float, default=0.8, help='Train-test split ratio for downstream task. (default: 0.8)')
    parser.add_argument('--train_test_random_seed', type=int, default=2025, help='Random seed for train-test split. (default: 2025)')

    parser.add_argument('--train_lr', type=float, default=0.0025, help='Learning rate for training. (default: 0.0025)')
    parser.add_argument('--train_eps', type=float, default=1e-7, help='Epsilon for Adam optimizer. (default: 1e-7)')
    parser.add_argument('--train_weight_decay', type=float, default=1e-15, help='Weight decay for Adam optimizer. (default: 1e-15)')

    parser.add_argument('--num_train_epoch', type=int, default=10, help='Number of training epochs. (default: 10)')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Batch size for training. (default: 4)')
    parser.add_argument('--train_num_workers', type=int, default=0, help='Number of workers to load data. (default: 0)')

    parser.add_argument('--train_internal_encoder_layers', type=int, default=1, help='Number of layers for internal encoder. (default: 1)')
    parser.add_argument('--train_encoder_layers', type=int, default=3, help='Number of layers for encoder. (default: 3)')
    parser.add_argument('--train_encoder_activation', type=str, default='leaky_relu', help='Activation function for encoder. (default: leaky_relu)')

    parser.add_argument('--pre_input_dim', type=int, default=8, help='Input feature dimension for pretraining. (default: 8)')
    parser.add_argument('--train_input_dim', type=int, default=1, help='Input feature dimension for training. (default: 1)')
    parser.add_argument('--train_hidden_dim', type=int, default=8, help='Hidden feature dimension for training. (default: 8)')
    parser.add_argument('--train_output_dim', type=int, default=8, help='Output feature dimension for training. (default: 8)')

    parser.add_argument('--train_result_folder', type=str, default='Results', help='Path to save training results. (default: Results)')
    parser.add_argument('--model_name', type=str, default='CellTOSG-Class', help='Model name. (default: CellTOSG-Class)')

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

    args.tissue = None
    args.suspension_type = None
    args.cell_type = None
    args.gender = None

    # Load dataset with conditions
    dataset = CellTOSGDataLoader(
        root=args.data_root,
        conditions={
            "tissue_general": args.tissue_general,
            # "tissue": args.tissue,
            # "suspension_type": args.suspension_type,
            # "cell_type": args.cell_type,
            "disease": args.disease_name,
            # "gender": args.gender,
        },
        downstream_task=args.downstream_task,  
        label_column=args.label_column,
        sample_ratio=args.sample_ratio,
        sample_size=args.sample_size,
        balanced=args.balanced,
        shuffle=args.shuffle,
        random_state=args.random_state,
        train_text=args.train_text,
        train_bio=args.train_bio,
        output_dir=args.dataset_output_dir
    )

    # Build Pretrain Model
    pretrain_model = build_pretrain_model(args, device)
    pretrain_model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))
    pretrain_model.eval()
    # Prepare text and seq embeddings
    x_name_emb, x_desc_emb, x_bio_emb = pre_embed_text(args, dataset, pretrain_model, device)
    # Graph feature
    xAll = dataset.data
    yAll = dataset.labels
    all_edge_index = dataset.edge_index
    internal_edge_index = dataset.internal_edge_index
    ppi_edge_index = dataset.ppi_edge_index
    # load embeddings into torch tensor
    xAll = torch.from_numpy(xAll).float().to(device)
    yAll = torch.from_numpy(yAll).long().to(device)
    x_name_emb = torch.from_numpy(x_name_emb).float().to(device)
    x_desc_emb = torch.from_numpy(x_desc_emb).float().to(device)
    x_bio_emb = torch.from_numpy(x_bio_emb).float().to(device)
    all_edge_index = torch.from_numpy(all_edge_index).long()
    internal_edge_index = torch.from_numpy(internal_edge_index).long()
    ppi_edge_index = torch.from_numpy(ppi_edge_index).long()
    
    # Split dataset into train and test for xAll and yAll
    xTr, xTe, yTr, yTe, num_classes = split_dataset(xAll, yAll, args, output_dir)
    args.num_class = num_classes
    # Build model
    model = build_model(args, device)
    # Train the model
    train(args, pretrain_model, model, device, xTr, xTe, yTr, yTe, all_edge_index, internal_edge_index, ppi_edge_index, x_name_emb, x_desc_emb, x_bio_emb)