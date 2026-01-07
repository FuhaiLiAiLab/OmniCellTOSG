import os
import pandas as pd
import numpy as np

from datetime import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

# custom dataloader
from GeoDataLoader.read_geograph import read_batch
from GeoDataLoader.geograph_sampler import GeoGraphLoader

# custom modules
from CellTOSG_Foundation.lm_model import TextEncoder, RNAGPT_LM, ProtGPT_LM
from CellTOSG_Downstream.decoder import CellTOSG_Class, DownGNNEncoder
from CellTOSG_Foundation.utils import tab_printer
from CellTOSG_Foundation.model import CellTOSG_Foundation, DegreeDecoder, EdgeDecoder, GNNEncoder
from CellTOSG_Foundation.mask import MaskEdge
from CellTOSG_Foundation.lm_model import TextEncoder, RNAGPT_LM, ProtGPT_LM
from CellTOSG_Loader import CellTOSGDataLoader

# Config loading
from utils import load_and_merge_configs, save_updated_config


def build_pretrain_model(args, device):
    # Build the mask for edge reconstruction
    mask = MaskEdge(p=args.p)
    # Build the text, rna and protein sequence encoders
    text_encoder = TextEncoder(args.text_lm_model_path, device)
    rna_seq_encoder = RNAGPT_LM(args.rna_seq_lm_model_path, args.rna_model_name, device)
    prot_seq_encoder = ProtGPT_LM(args.prot_model_name, device)
    # Build the internal GNN encoder, graph GNN encoder
    internal_graph_encoder = GNNEncoder(args.num_omic_feature, args.pre_internal_input_dim, args.pre_internal_output_dim,
                            num_layers=args.pre_internal_encoder_layers, dropout=args.pre_internal_encoder_dropout,
                            bn=args.pre_internal_bn, layer=args.pre_internal_layer_type, activation=args.pre_internal_encoder_activation)
    graph_encoder = GNNEncoder(args.num_omic_feature, args.pre_graph_input_dim, args.pre_graph_output_dim,
                        num_layers=args.pre_graph_encoder_layers, dropout=args.pre_graph_encoder_dropout,
                        bn=args.pre_graph_bn, layer=args.pre_graph_layer_type, activation=args.pre_graph_encoder_activation)
    # Build the edge and degree decoder
    edge_decoder = EdgeDecoder(args.pre_graph_output_dim, args.pre_decoder_dim,
                            num_layers=args.pre_decoder_layers, dropout=args.pre_decoder_dropout)
    degree_decoder = DegreeDecoder(args.pre_graph_output_dim, args.pre_decoder_dim,
                                num_layers=args.pre_decoder_layers, dropout=args.pre_decoder_dropout)
    # Build the pretraining model
    pretrain_model = CellTOSG_Foundation(
                    num_entity=args.num_entity,
                    text_input_dim=args.pre_lm_emb_dim,
                    omic_input_dim=args.num_omic_feature,
                    cross_fusion_output_dim=args.pre_cross_fusion_output_dim, 
                    text_encoder=text_encoder,
                    rna_seq_encoder=rna_seq_encoder,
                    prot_seq_encoder=prot_seq_encoder,
                    encoder=graph_encoder,
                    internal_encoder=internal_graph_encoder,
                    edge_decoder=edge_decoder,
                    degree_decoder=degree_decoder,
                    mask=mask,
                    entity_mlp_dims=args.entity_mlp_dims,
                    ).to(device)
    return pretrain_model


def build_model(args, device):
    # Build the internal GNN encoder, graph GNN encoder
    internal_graph_encoder = DownGNNEncoder(args.train_internal_input_dim, args.train_internal_hidden_dim, args.train_internal_output_dim,
                            num_layers=args.train_internal_encoder_layers, dropout=args.train_internal_encoder_dropout,
                            bn=args.train_internal_bn, layer=args.train_internal_layer_type, activation=args.train_internal_encoder_activation)
    graph_encoder = DownGNNEncoder(args.train_graph_input_dim, args.train_graph_hidden_dim, args.train_graph_output_dim,
                    num_layers=args.train_graph_encoder_layers, dropout=args.train_graph_encoder_dropout,
                    bn=args.train_graph_bn, layer=args.train_graph_layer_type, activation=args.train_graph_encoder_activation)
    # Build the downstream task model
    model = CellTOSG_Class(text_input_dim=args.train_lm_emb_dim,
                    omic_input_dim=args.num_omic_feature,
                    cross_fusion_output_dim=args.train_cross_fusion_output_dim,
                    pre_input_output_dim=args.pre_input_output_dim,
                    final_fusion_output_dim=args.final_fusion_output_dim,
                    num_class=args.num_class,
                    num_entity=args.num_entity,
                    linear_hidden_dims=args.train_linear_hidden_dims,
                    linear_activation=args.train_linear_activation,
                    linear_dropout_rate=args.train_linear_dropout,
                    encoder=graph_encoder,
                    internal_encoder=internal_graph_encoder,
                    entity_mlp_dims=args.entity_mlp_dims,  # num_entity → decrease → increase → num_entity
                    mlp_dropout=0.1,
                    ).to(device)
    return model


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
        x_name_emb = text_encoder.generate_embeddings(name_sentence_list, batch_size=args.pretrain_text_batch_size, text_emb_dim=args.train_lm_emb_dim)
        x_desc_emb = text_encoder.generate_embeddings(desc_sentence_list, batch_size=args.pretrain_text_batch_size, text_emb_dim=args.train_lm_emb_dim)
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
        rna_seq_embeddings = rna_seq_encoder.generate_embeddings(rna_replaced_seq_list, batch_size=args.pretrain_text_batch_size, max_len=args.rna_seq_max_len, seq_emb_dim=args.train_lm_emb_dim)
        # sequence list where type == protein
        prot_seq_list = s_bio[s_bio['Type'] == 'Protein']['Sequence'].tolist()
        prot_seq_encoder = pretrain_model.prot_seq_encoder
        prot_seq_encoder.load_model()
        prot_replaced_seq_list = ['X' if type(i) == float else i for i in prot_seq_list]
        prot_seq_embeddings = prot_seq_encoder.generate_embeddings(prot_replaced_seq_list, seq_emb_dim=args.train_lm_emb_dim)
        x_bio_emb = torch.cat((rna_seq_embeddings, prot_seq_embeddings), dim=0)
    else:
        # Use pre-computed embeddings
        x_bio_emb = dataset.x_bio_emb

    return x_name_emb, x_desc_emb, x_bio_emb

def get_num_classes(
    yTr: torch.Tensor,
    yTe: torch.Tensor,
) -> int:

    yTr = yTr.view(-1)
    yTe = yTe.view(-1)

    all_y = torch.cat([yTr, yTe], dim=0)
    unique_values = torch.unique(all_y)
    unique_values, _ = torch.sort(unique_values)

    num_classes = int(unique_values.numel())

    train_num_cell = int(yTr.numel())
    test_num_cell = int(yTe.numel())

    print(f"\nUnique label values in (train+test): {unique_values}")
    print(f"Number of classes: {num_classes}")

    print("\nTraining set class distribution:")
    for lab in unique_values:
        class_count = torch.sum(yTr == lab).item()
        percentage = (class_count / train_num_cell) * 100 if train_num_cell > 0 else 0.0
        print(f"Class {int(lab.item())}: {class_count} samples ({percentage:.1f}%)")

    print("\nTesting set class distribution:")
    for lab in unique_values:
        class_count = torch.sum(yTe == lab).item()
        percentage = (class_count / test_num_cell) * 100 if test_num_cell > 0 else 0.0
        print(f"Class {int(lab.item())}: {class_count} samples ({percentage:.1f}%)")

    return num_classes

def write_best_model_info(path, max_test_acc_id, epoch_loss_list, epoch_acc_list, epoch_f1_list, test_loss_list, test_acc_list, test_f1_list):
    best_model_info = (
        f'\n-------------BEST TEST ACCURACY MODEL ID INFO: {max_test_acc_id} -------------\n'
        '--- TRAIN ---\n'
        f'BEST MODEL TRAIN LOSS: {epoch_loss_list[max_test_acc_id - 1]}\n'
        f'BEST MODEL TRAIN ACCURACY: {epoch_acc_list[max_test_acc_id - 1]}\n'
        f'BEST MODEL TRAIN F1 SCORE: {epoch_f1_list[max_test_acc_id - 1]}\n'
        '--- TEST ---\n'
        f'BEST MODEL TEST LOSS: {test_loss_list[max_test_acc_id - 1]}\n'
        f'BEST MODEL TEST ACCURACY: {test_acc_list[max_test_acc_id - 1]}\n'
        f'BEST MODEL TEST F1 SCORE: {test_f1_list[max_test_acc_id - 1]}\n'
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
        # ********************** MLP Information Flow *************************
        # Reshape: (batch*num_entity, feature) → (batch, num_entity)
        z = z.view(current_cell_num, num_entity, -1).squeeze(-1)
        # Apply MLP on entity dimension
        for mlp in pretrain_model.entity_mlp_layers:
            z = mlp(z)
        z = pretrain_model.layer_norm(z)
        # Expand and flatten: (batch, num_entity) → (batch*num_entity, 1)
        z = z.reshape(-1, 1)
        # **********************************************************************
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
        # ********************** MLP Information Flow *************************
        # Reshape: (batch*num_entity, feature) → (batch, num_entity)
        z = z.view(current_cell_num, num_entity, -1).squeeze(-1)
        # Apply MLP on entity dimension
        for mlp in pretrain_model.entity_mlp_layers:
            z = mlp(z)
        z = pretrain_model.layer_norm(z)
        # Expand and flatten: (batch, num_entity) → (batch*num_entity, 1)
        z = z.reshape(-1, 1)
        # **********************************************************************
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


def train(args, pretrain_model, model, device, xTr, xTe, yTr, yTe, all_edge_index, internal_edge_index, ppi_edge_index, x_name_emb, x_desc_emb, x_bio_emb, config_groups):
    train_num_cell = xTr.shape[0]
    num_entity = xTr.shape[1]
    num_feature = args.num_omic_feature
    epoch_num = args.num_train_epoch
    train_batch_size = args.train_batch_size
    learning_rate = args.train_lr
    random_state = args.random_state

    epoch_loss_list = []
    epoch_acc_list = []
    epoch_f1_list = []
    
    test_loss_list = []
    test_acc_list = []
    test_f1_list = []

    max_test_acc = 0
    max_test_acc_id = 0

    # Clean result previous epoch_i_pred files
    folder_name = 'epoch_' + str(epoch_num) + '_' + str(train_batch_size) + '_' + str(learning_rate) + '_' + str(random_state)

    #Add timestamp to folder name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    base_path = os.path.join(
        '.', args.train_result_folder,
        args.task,
        args.disease_name,
        args.train_base_layer
    )

    os.makedirs(base_path, exist_ok=True)

    path = os.path.join(base_path, f"{folder_name}_{timestamp}")
    os.mkdir(path)

    # Save final configuration for reference
    config_save_path = os.path.join(path, 'config.yaml')
    save_updated_config(config_groups, config_save_path)
    print(f"[Config] Saved to {config_save_path}")

    # np.save(os.path.join(path, "xTr.npy"), xTr.cpu().numpy())
    # np.save(os.path.join(path, "xTe.npy"), xTe.cpu().numpy())
    # np.save(os.path.join(path, "yTr.npy"), yTr.cpu().numpy())
    # np.save(os.path.join(path, "yTe.npy"), yTe.cpu().numpy())
    # print(f"[Data Split] Saved train/test splits to {path}")

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
        score_list = yTr.detach().cpu().numpy().reshape(-1).tolist()
        epoch_ypred_list = epoch_ypred.reshape(-1).tolist()
        train_dict = {'label': score_list, 'prediction': epoch_ypred_list}
        tmp_training_input_df = pd.DataFrame(train_dict)
        # Calculating metrics
        accuracy = accuracy_score(tmp_training_input_df['label'], tmp_training_input_df['prediction'])
        tmp_training_input_df.to_csv(path + '/TrainingPred_' + str(i) + '.txt', index=False, header=True)
        epoch_acc_list.append(accuracy)

        train_f1 = f1_score(
            tmp_training_input_df["label"],
            tmp_training_input_df["prediction"],
            average="macro",
            zero_division=0,
        )

        epoch_f1_list.append(train_f1)


        conf_matrix = confusion_matrix(tmp_training_input_df['label'], tmp_training_input_df['prediction'])
        print('EPOCH ' + str(i) + ' TRAINING ACCURACY: ', accuracy)
        print('EPOCH ' + str(i) + ' TRAINING F1 SCORE: ', train_f1)
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
        
        test_f1 = f1_score(
            tmp_test_input_df["label"],
            tmp_test_input_df["prediction"],
            average="macro",
            zero_division=0,
        )
        test_f1_list.append(test_f1)

        print('\n-------------EPOCH TEST ACCURACY LIST: -------------')
        print(test_acc_list)
        print('\n-------------EPOCH TEST MSE LOSS LIST: -------------')
        print(test_loss_list)

        if args.use_wandb:
            wandb.log({
                "epoch": i,
                "train_loss": epoch_loss_list[-1],
                "train_acc": epoch_acc_list[-1],
                "train_f1": epoch_f1_list[-1],
                "test_loss": test_loss_list[-1],
                "test_acc": test_acc_list[-1],
                "test_f1": test_f1_list[-1],
            })

        # SAVE BEST TEST MODEL
        if test_acc >= max_test_acc:
            max_test_acc = test_acc
            max_test_acc_id = i
            # torch.save(model.state_dict(), path + '/best_train_model'+ str(i) +'.pt')
            torch.save(model.state_dict(), path + '/best_train_model.pt')
            tmp_training_input_df.to_csv(path + '/BestTrainingPred.txt', index=False, header=True)
            tmp_test_input_df.to_csv(path + '/BestTestPred.txt', index=False, header=True)
            write_best_model_info(path, max_test_acc_id, epoch_loss_list, epoch_acc_list, epoch_f1_list, test_loss_list, test_acc_list, test_f1_list)
        print('\n-------------BEST TEST ACCURACY MODEL ID INFO:' + str(max_test_acc_id) + '-------------')
        print('--- TRAIN ---')
        print('BEST MODEL TRAIN LOSS: ', epoch_loss_list[max_test_acc_id - 1])
        print('BEST MODEL TRAIN ACCURACY: ', epoch_acc_list[max_test_acc_id - 1])
        print('BEST MODEL TRAIN F1 SCORE: ', epoch_f1_list[max_test_acc_id - 1])
        print('--- TEST ---')
        print('BEST MODEL TEST LOSS: ', test_loss_list[max_test_acc_id - 1])
        print('BEST MODEL TEST ACCURACY: ', test_acc_list[max_test_acc_id - 1])
        print('BEST MODEL TEST F1 SCORE: ', test_f1_list[max_test_acc_id - 1])


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
    score_list = yTe.detach().cpu().numpy().reshape(-1).tolist()
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


if __name__ == "__main__":
    # Load and merge configurations with command line override support
    args, config_groups = load_and_merge_configs(
        'Configs/dataloader.yaml',
        'Configs/pretraining.yaml',
        'Configs/training.yaml'
    )

    args.pretrained_model_save_path = args.pretrained_model_save_path.format(
        pretrain_base_layer=args.pretrain_base_layer
    )

    # Set internal and graph layer types to the same base layer type
    args.train_internal_layer_type = args.train_base_layer
    args.train_graph_layer_type = args.train_base_layer

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

    # Use extracted data if available
    from pathlib import Path    
    args.use_extracted_data = True

    data_dir = Path(args.dataset_output_dir)
    required_files = ["train/expression_matrix.npy", "train/labels_train.npy", "test/expression_matrix.npy", "test/labels_test.npy"]

    def all_required_files_exist(path, filenames):
        return all((path / f).exists() for f in filenames)

    if args.use_extracted_data and all_required_files_exist(data_dir, required_files):
        print("[Info] Using extracted data from:", data_dir)

        class FixedDataset:
            def __init__(self, dataset_root, dataset_output_dir):
                dataset_output_dir = Path(dataset_output_dir)
                dataset_root = Path(dataset_root)

                self.x_train = np.load(f"{dataset_output_dir}/train/expression_matrix.npy")
                self.x_test = np.load(f"{dataset_output_dir}/test/expression_matrix.npy")

                self.y_train = np.load(f"{dataset_output_dir}/train/labels_train.npy")
                self.y_test = np.load(f"{dataset_output_dir}/test/labels_test.npy")

                self.edge_index = np.load(dataset_root / "edge_index.npy")
                self.internal_edge_index = np.load(dataset_root / "internal_edge_index.npy")
                self.ppi_edge_index = np.load(dataset_root / "ppi_edge_index.npy")
                self.x_name_emb = np.load(dataset_root / "x_name_emb.npy")
                self.x_desc_emb = np.load(dataset_root / "x_desc_emb.npy")
                self.x_bio_emb = np.load(dataset_root / "x_bio_emb.npy")

        dataset = FixedDataset(args.dataset_root, args.dataset_output_dir)

        # Graph feature
        xTr = dataset.x_train
        xTe = dataset.x_test
        yTr = dataset.y_train
        yTe = dataset.y_test

    else:
        if not data_dir.exists():
            print(f"[Info] Output directory '{data_dir}' not found. It will be created.")
        else:
            missing = [f for f in required_files if not (data_dir / f).exists()]
            print(f"[Info] Missing files in extracted data: {missing}. Running data extraction.")

        print("[Info] Running CellTOSGDataLoader to extract data...")

        # Load dataset with conditions
        dataset = CellTOSGDataLoader(
            root=args.dataset_root,
            conditions={
                "tissue_general": args.tissue_general,
                # "tissue": args.tissue,
                # "suspension_type": args.suspension_type,
                # "cell_type": args.cell_type,
                "disease": args.disease_name,
                # "gender": args.gender,
            },
            task=args.task,  
            label_column=args.label_column,
            sample_ratio=args.sample_ratio,
            sample_size=args.sample_size,
            shuffle=args.shuffle,
            stratified_balancing=args.stratified_balancing,
            extract_mode=args.extract_mode,
            random_state=args.random_state,
            train_text=args.train_text,
            train_bio=args.train_bio,
            correction_method=args.correction_method,
            output_dir=args.dataset_output_dir
        )

        # Graph feature
        X = dataset.data            # dict: {"train": X_train, "test": X_test}
        Y = dataset.labels          # dict: {"train": y_train, "test": y_test}
        metadata = dataset.metadata # dict: {"train": df_train, "test": df_test}

        xTr = X["train"]
        xTe = X["test"]
        yTr = Y["train"]
        yTe = Y["test"]

    # Replace spaces and quotes in disease name after loading the dataset
    args.disease_name = args.disease_name.replace("'", "").replace(" ", "_")

    args.use_wandb = True

    if args.use_wandb:
        import wandb
        wandb.init(
            project=f"{args.task}-celltosg",
            name=f"{args.task}_{args.disease_name}_{args.train_base_layer}_bs{args.train_batch_size}_lr{args.train_lr}_rs{args.random_state}",
            config=vars(args)
        )


    print(f"Number of training cells: {xTr.shape[0]}")
    print(f"Number of testing cells: {xTe.shape[0]}")

    # Build Pretrain Model
    args.num_entity = xTr.shape[1]
    print(f"Number of entities: {args.num_entity}")
    pretrain_model = build_pretrain_model(args, device)
    pretrain_model.load_state_dict(torch.load(args.pretrained_model_save_path, map_location=device))
    pretrain_model.eval()

    all_edge_index = dataset.edge_index
    internal_edge_index = dataset.internal_edge_index
    ppi_edge_index = dataset.ppi_edge_index

    # Prepare text and seq embeddings
    x_name_emb, x_desc_emb, x_bio_emb = pre_embed_text(args, dataset, pretrain_model, device)
    x_name_emb = torch.from_numpy(x_name_emb).float().to(device)
    x_desc_emb = torch.from_numpy(x_desc_emb).float().to(device)
    x_bio_emb = torch.from_numpy(x_bio_emb).float().to(device)

    # load graph into torch tensor
    xTr = torch.from_numpy(xTr).float().to(device)
    xTe = torch.from_numpy(xTe).float().to(device)
    yTr = torch.from_numpy(yTr).long().view(-1, 1).to(device)
    yTe = torch.from_numpy(yTe).long().view(-1, 1).to(device)

    all_edge_index = torch.from_numpy(all_edge_index).long()
    internal_edge_index = torch.from_numpy(internal_edge_index).long()
    ppi_edge_index = torch.from_numpy(ppi_edge_index).long()
    
    # get number of classes
    args.num_class = get_num_classes(yTr, yTe)
 
    # Build model
    model = build_model(args, device)

    # Train the model
    train(args, pretrain_model, model, device, xTr, xTe, yTr, yTe, all_edge_index, internal_edge_index, ppi_edge_index, x_name_emb, x_desc_emb, x_bio_emb, config_groups)

# python train.py --train_lr 0.0005 --train_batch_size 4 --train_base_layer gat --task cell_type --label_column cell_type --tissue_general brain --disease_name "Alzheimer's Disease" --sample_ratio 0.1 --dataset_output_dir ./Data/train_ad_celltype_0.1_42 --random_state 42

# python train.py --train_lr 0.0005 --train_batch_size 4 --train_base_layer gat --task disease --label_column disease --tissue_general brain --disease_name "Alzheimer's Disease" --sample_ratio 0.1 --dataset_output_dir ./Data/train_ad_disease_0.1_42 --random_state 42