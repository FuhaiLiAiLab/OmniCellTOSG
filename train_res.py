import os
import pandas as pd
import numpy as np

from datetime import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

# custom dataloader
from GeoDataLoader.read_geograph import read_batch, read_drug_batch
from GeoDataLoader.geograph_sampler import GeoGraphLoader

# custom modules
from CellTOSG_Foundation.lm_model import TextEncoder, RNAGPT_LM, ProtGPT_LM
from CellTOSG_Downstream.decoder import CellTOSG_DrugRes, DownGNNEncoder
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
    pretrain_model = CellTOSG_Foundation(num_entity=args.num_entity,
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
                    entity_mlp_dims=[32,32], #######################################################################
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
    ####################################################################### Need to add all drug encoder parameters here ####################
    drug_encoder = DownGNNEncoder(args.drug_input_dim, args.train_drug_hidden_dim, args.train_drug_output_dim,
                    num_layers=args.train_drug_encoder_layers, dropout=args.train_drug_encoder_dropout,
                    bn=args.train_drug_bn, layer=args.train_drug_layer_type, activation=args.train_drug_encoder_activation)
    ####################################################################### Need to add all drug encoder parameters here ####################
    # Build the downstream task model
    model = CellTOSG_DrugRes(text_input_dim=args.train_lm_emb_dim,
                        omic_input_dim=args.num_omic_feature,
                        drug_output_dim=args.train_drug_output_dim, #################################################################################################
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
                        drug_encoder=drug_encoder,
                        num_drug_per_point=args.num_drug_per_point,
                        entity_mlp_dims=[32, 32],  # num_entity → decrease → increase → num_entity
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


def split_dataset(xAll, yAll, args):
    """
    Split dataset into train and test sets ensuring relatively even distribution for each class.
    Maps both numerical and textual values to vectorized values and saves mapping dictionary.
    
    Args:
        xAll: Input features tensor
        yAll: Labels tensor
        args: Command line arguments containing split parameters
    
    Returns:
        tuple: (xTr, xTe, yTr, yTe, num_entity, num_feature)
    """    
    # Get unique values from yAll
    yAll = yAll.view(-1, 1)
    unique_values = torch.unique(yAll)
    num_classes = len(unique_values)
    print("\n")
    print(f"Original unique values: {unique_values}")
    print(f"Total number of classes: {num_classes}")
    
    # Print class distribution
    print("\nOriginal dataset class distribution:")
    for class_idx in range(num_classes):
        class_count = torch.sum(yAll == class_idx)
        percentage = (class_count / len(yAll)) * 100
        print(f"Class {class_idx}: {class_count} samples ({percentage:.1f}%)")

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
        print(f"Class {class_idx}: {class_count} samples ({percentage:.1f}%)")
    print("\nTesting set class distribution:")
    for class_idx in range(num_classes):
        class_count = torch.sum(yTe == class_idx)
        percentage = (class_count / test_num_cell) * 100
        print(f"Class {class_idx}: {class_count} samples ({percentage:.1f}%)")

    return xTr, xTe, yTr, yTe, num_classes

def write_best_model_info(path, max_test_epoch_id, epoch_loss_list, epoch_pearson_list,
                          test_loss_list, test_pearson_list):
    """
    Save information about the best model (based on test Pearson) to a text file.
    """
    # epoch indices in lists are 0-based, while epoch ids are 1-based
    idx = max_test_epoch_id - 1

    best_model_info = (
        f'\n-------------BEST TEST PEARSON MODEL ID INFO: {max_test_epoch_id} -------------\n'
        '--- TRAIN ---\n'
        f'BEST MODEL TRAIN LOSS: {epoch_loss_list[idx]}\n'
        f'BEST MODEL TRAIN PEARSON: {epoch_pearson_list[idx]}\n'
        '--- TEST ---\n'
        f'BEST MODEL TEST LOSS: {test_loss_list[idx]}\n'
        f'BEST MODEL TEST PEARSON: {test_pearson_list[idx]}\n'
    )
    with open(os.path.join(path, 'best_model_info.txt'), 'w') as file:
        file.write(best_model_info)


def train_model(train_dataset_loader, current_cell_num, num_entity, name_embeddings, desc_embeddings, seq_embeddings,
                train_drug_dataset_loader, pretrain_model, model, device, args):
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.train_lr, eps=args.train_eps, weight_decay=args.train_weight_decay)
    batch_loss = 0
    for batch_idx, (cell_data, drug_data) in enumerate(zip(train_dataset_loader, train_drug_dataset_loader)):
        optimizer.zero_grad()
        x = Variable(cell_data.x.float(), requires_grad=False).to(device)
        internal_edge_index = Variable(cell_data.internal_edge_index, requires_grad=False).to(device)
        ppi_edge_index = Variable(cell_data.edge_index, requires_grad=False).to(device)
        edge_index = Variable(cell_data.all_edge_index, requires_grad=False).to(device)
        label = Variable(cell_data.label, requires_grad=False).to(device)
        
        # Extract drug data
        drug_chem_x = Variable(drug_data.x.float(), requires_grad=False).to(device)
        drug_chem_edge_index = Variable(drug_data.edge_index, requires_grad=False).to(device)
        drug_batch = drug_data.batch.to(device)  # This indicates which graph each atom belongs to

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
        output = model(x, pre_x, edge_index, internal_edge_index, ppi_edge_index,
                            num_entity, drug_chem_x, drug_chem_edge_index, drug_batch, args.num_drug_per_point,
                            name_embeddings, desc_embeddings, seq_embeddings, current_cell_num)
        loss = model.loss(output, label)
        loss.backward()
        batch_loss += loss.item()
        print('Label: ', label)
        print('Prediction: ', output)
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        # # check pretrain model parameters
        # state_dict = pretrain_model.internal_encoder.state_dict()
        # print(state_dict['convs.1.lin.weight'])
        # print(model.embedding.weight.data)
    torch.cuda.empty_cache()
    return model, batch_loss, output


def test_model(test_dataset_loader, current_cell_num, num_entity, test_drug_dataset_loader, name_embeddings, desc_embeddings, seq_embeddings, pretrain_model, model, device, args):
    batch_loss = 0
    for batch_idx, (cell_data, drug_data) in enumerate(zip(test_dataset_loader, test_drug_dataset_loader)):
        x = Variable(cell_data.x.float(), requires_grad=False).to(device)
        internal_edge_index = Variable(cell_data.internal_edge_index, requires_grad=False).to(device)
        ppi_edge_index = Variable(cell_data.edge_index, requires_grad=False).to(device)
        edge_index = Variable(cell_data.all_edge_index, requires_grad=False).to(device)
        label = Variable(cell_data.label, requires_grad=False).to(device)

        # Extract drug data
        drug_chem_x = Variable(drug_data.x.float(), requires_grad=False).to(device)
        drug_chem_edge_index = Variable(drug_data.edge_index, requires_grad=False).to(device)
        drug_batch = drug_data.batch.to(device)  # This indicates which graph each atom belongs to

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
        output = model(x, pre_x, edge_index, internal_edge_index, ppi_edge_index,
                            num_entity, drug_chem_x, drug_chem_edge_index, drug_batch, args.num_drug_per_point,
                            name_embeddings, desc_embeddings, seq_embeddings, current_cell_num)
        loss = model.loss(output, label)
        batch_loss += loss.item()
        print('Label: ', label)
        print('Prediction: ', output)
    return model, batch_loss, output


def train(args, pretrain_model, model, device, xTr, xTe, yTr, yTe, all_edge_index, internal_edge_index, ppi_edge_index, 
          drug_index_Tr, drug_index_Te, x_name_emb, x_desc_emb, x_bio_emb, config_groups):
    train_num_cell = xTr.shape[0]
    num_entity = xTr.shape[1]
    num_feature = args.num_omic_feature
    epoch_num = args.num_train_epoch
    train_batch_size = args.train_batch_size
    learning_rate = args.train_lr
    train_test_random_seed = args.train_test_random_seed

    epoch_loss_list = []
    epoch_pearson_list = []
    test_loss_list = []
    test_pearson_list = []
    max_test_pearson = 0
    max_test_pearson_id = 0

    # Clean result previous epoch_i_pred files
    folder_name = 'epoch_' + str(epoch_num) + '_' + str(train_batch_size) + '_' + str(learning_rate) + '_' + str(train_test_random_seed)

    #Add timestamp to folder name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    base_path = os.path.join(
        '.', args.train_result_folder,
        args.downstream_task,
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

    np.save(os.path.join(path, "xTr.npy"), xTr.cpu().numpy())
    np.save(os.path.join(path, "xTe.npy"), xTe.cpu().numpy())
    np.save(os.path.join(path, "yTr.npy"), yTr.cpu().numpy())
    np.save(os.path.join(path, "yTe.npy"), yTe.cpu().numpy())
    print(f"[Data Split] Saved train/test splits to {path}")

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
            geo_train_drug_datalist = read_drug_batch(index, upper_index, drug_index_Tr)
            train_drug_dataset_loader = GeoGraphLoader.load_graph(geo_train_drug_datalist, args.train_batch_size * args.num_drug_per_point, args.train_num_workers)

            current_cell_num = upper_index - index # current batch size
            model, batch_loss, batch_ypred = train_model(train_dataset_loader, current_cell_num, num_entity, x_name_emb, x_desc_emb, x_bio_emb,
                                                                    train_drug_dataset_loader, pretrain_model, model, device, args)
            print('BATCH LOSS: ', batch_loss)
            batch_loss_list.append(batch_loss)
            # PRESERVE PREDICTION OF BATCH TRAINING DATA
            batch_ypred = (Variable(batch_ypred).data).cpu().numpy().reshape(-1, 1)
            epoch_ypred = np.vstack((epoch_ypred, batch_ypred))
        epoch_loss = np.mean(batch_loss_list)
        print('TRAIN EPOCH ' + str(i) + ' LOSS: ', epoch_loss)
        epoch_loss_list.append(epoch_loss)
        epoch_ypred = np.delete(epoch_ypred, 0, axis = 0)
        # print('ITERATION NUMBER UNTIL NOW: ' + str(iteration_num))
        # Preserve corr for every epoch
        score_list = yTr.detach().cpu().numpy().reshape(-1).tolist()
        epoch_ypred_list = epoch_ypred.reshape(-1).tolist()
        train_dict = {'label': score_list, 'prediction': epoch_ypred_list}
        tmp_training_input_df = pd.DataFrame(train_dict)
        # Calculating metrics
        tmp_training_input_df.to_csv(path + '/TrainingPred_' + str(i) + '.txt', index=False, header=True)
        epoch_pearson = tmp_training_input_df.corr(method='pearson')
        epoch_pearson_list.append(epoch_pearson['prediction']['label'])
        tmp_training_input_df.to_csv(path + '/TrainingPred_' + str(i) + '.txt', index=False, header=True)
        print('EPOCH ' + str(i) + ' PEARSON CORRELATION: ', epoch_pearson)
        print('\n-------------EPOCH TRAINING PEARSON CORRELATION LIST: -------------')
        print(epoch_pearson_list)
        print('\n-------------EPOCH TRAINING LOSS LIST: -------------')
        print(epoch_loss_list)

        # # # Test model on test dataset
        test_pearson, test_loss, tmp_test_input_df = test(args, pretrain_model, model, xTe, yTe, all_edge_index, internal_edge_index, ppi_edge_index, drug_index_Te, x_name_emb, x_desc_emb, x_bio_emb, device, i)
        test_pearson_list.append(test_pearson['prediction']['label'])
        test_loss_list.append(test_loss)
        tmp_test_input_df.to_csv(path + '/TestPred' + str(i) + '.txt', index=False, header=True)
        print('\n-------------EPOCH TEST PEARSON CORRELATION LIST: -------------')
        print(test_pearson_list)
        print('\n-------------EPOCH TEST MSE LOSS LIST: -------------')
        print(test_loss_list)

        if args.use_wandb:
            wandb.log({
                "epoch": i,
                "train_loss": epoch_loss_list[-1],
                "train_pearson": epoch_pearson_list[-1],
                "test_loss": test_loss_list[-1],
                "test_pearson": test_pearson_list[-1]
            })

        # SAVE BEST TEST MODEL
        test_pearson = test_pearson['prediction']['label']
        if test_pearson >= max_test_pearson:
            max_test_pearson = test_pearson
            max_test_pearson_id = i
            # torch.save(model.state_dict(), path + '/best_train_model'+ str(i) +'.pt')
            torch.save(model.state_dict(), path + '/best_train_model.pt')
            tmp_training_input_df.to_csv(path + '/BestTrainingPred.txt', index=False, header=True)
            tmp_test_input_df.to_csv(path + '/BestTestPred.txt', index=False, header=True)
            write_best_model_info(
                path,
                max_test_pearson_id,
                epoch_loss_list,
                epoch_pearson_list,
                test_loss_list,
                test_pearson_list,
            )
        print('\n-------------BEST TEST PEARSON CORRELATION MODEL ID INFO:' + str(max_test_pearson_id) + '-------------')
        print('--- TRAIN ---')
        print('BEST MODEL TRAIN LOSS: ', epoch_loss_list[max_test_pearson_id - 1])
        print('BEST MODEL TRAIN PEARSON CORRELATION: ', epoch_pearson_list[max_test_pearson_id - 1])
        print('--- TEST ---')
        print('BEST MODEL TEST LOSS: ', test_loss_list[max_test_pearson_id - 1])
        print('BEST MODEL TEST PEARSON CORRELATION: ', test_pearson_list[max_test_pearson_id - 1])


def test(args, pretrain_model, model, xTe, yTe, all_edge_index, internal_edge_index, ppi_edge_index, drug_index_Te, x_name_emb, x_desc_emb, x_bio_emb, device, i):
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
        geo_train_drug_datalist = read_drug_batch(index, upper_index, drug_index_Te)
        test_drug_dataset_loader = GeoGraphLoader.load_graph(geo_train_drug_datalist, args.train_batch_size * args.num_drug_per_point, args.train_num_workers)

        print('TEST MODEL...')
        current_cell_num = upper_index - index # current batch size
        model, batch_loss, batch_ypred = test_model(test_dataset_loader, current_cell_num, num_entity, test_drug_dataset_loader, x_name_emb, x_desc_emb, x_bio_emb, pretrain_model, model, device, args)
        print('BATCH LOSS: ', batch_loss)
        batch_loss_list.append(batch_loss)
        # PRESERVE PREDICTION OF BATCH TEST DATA
        batch_ypred = (Variable(batch_ypred).data).cpu().numpy().reshape(-1, 1)
        all_ypred = np.vstack((all_ypred, batch_ypred))
    test_loss = np.mean(batch_loss_list)
    print('EPOCH ' + str(i) + ' TEST LOSS: ', test_loss)
    # Preserve pearson correlation for every epoch
    all_ypred = np.delete(all_ypred, 0, axis=0)
    all_ypred_lists = list(all_ypred)
    all_ypred_list = [item for elem in all_ypred_lists for item in elem]
    score_list = yTe.detach().cpu().numpy().reshape(-1).tolist()
    test_dict = {'label': score_list, 'prediction': all_ypred_list}
    # import pdb; pdb.set_trace()
    tmp_test_input_df = pd.DataFrame(test_dict)
    test_pearson = tmp_test_input_df.corr(method = 'pearson')
    print('PEARSON CORRELATION: ', test_pearson)
    return test_pearson, test_loss, tmp_test_input_df


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
    required_files = ["expression_matrix_corrected.npy", "labels.npy"]

    def all_required_files_exist(path, filenames):
        return all((path / f).exists() for f in filenames)

    if args.use_extracted_data and all_required_files_exist(data_dir, required_files):
        print("[Info] Using extracted data from:", data_dir)


        class FixedDataset:
            def __init__(self, dataset_root, dataset_output_dir):
                dataset_output_dir = Path(dataset_output_dir)
                dataset_root = Path(dataset_root)

                self.data = np.load(dataset_output_dir / "expression_matrix_corrected.npy")
                self.labels = np.load(dataset_output_dir / "labels.npy")
                self.edge_index = np.load(dataset_root / "edge_index.npy")
                self.internal_edge_index = np.load(dataset_root / "internal_edge_index.npy")
                self.ppi_edge_index = np.load(dataset_root / "ppi_edge_index.npy")
                self.x_name_emb = np.load(dataset_root / "x_name_emb.npy")
                self.x_desc_emb = np.load(dataset_root / "x_desc_emb.npy")
                self.x_bio_emb = np.load(dataset_root / "x_bio_emb.npy")

        dataset = FixedDataset(args.dataset_root, args.dataset_output_dir)

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

    # Replace spaces and quotes in disease name after loading the dataset
    args.disease_name = args.disease_name.replace("'", "").replace(" ", "_")

    args.use_wandb = True

    if args.use_wandb:
        import wandb
        wandb.init(
            project=f"{args.downstream_task}-celltosg",
            name=f"{args.downstream_task}_{args.disease_name}_{args.train_base_layer}_bs{args.train_batch_size}_lr{args.train_lr}_rs{args.train_test_random_seed}",
            config=vars(args)
        )

    # Graph feature
    xAll = dataset.data
    yAll = dataset.labels
    # Build Pretrain Model
    args.num_entity = xAll.shape[1]
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
    # load embeddings into torch tensor
    xAll = torch.from_numpy(xAll).float().to(device)
    yAll = torch.from_numpy(yAll).long().to(device)
    all_edge_index = torch.from_numpy(all_edge_index).long()
    internal_edge_index = torch.from_numpy(internal_edge_index).long()
    ppi_edge_index = torch.from_numpy(ppi_edge_index).long()
    
    # Split dataset into train and test for xAll and yAll
    xTr, xTe, yTr, yTe, num_classes = split_dataset(xAll, yAll, args)
    args.num_class = num_classes

    # Randomly generate drug indices (0-3) for training and testing sets
    num_drug_per_point = 2
    num_drugs = 4  # drugs indexed 0-3
    np.random.seed(args.train_test_random_seed)  # Use same random seed for reproducibility
    drug_index_Tr = np.random.randint(0, num_drugs, size=(yTr.shape[0], num_drug_per_point))
    drug_index_Te = np.random.randint(0, num_drugs, size=(yTe.shape[0], num_drug_per_point))
    print(f"\nDrug index generation:")
    print(f"Training set drug indices shape: {drug_index_Tr.shape}")
    print(f"Testing set drug indices shape: {drug_index_Te.shape}")
    print(f"Unique drugs in training: {np.unique(drug_index_Tr)}")
    print(f"Unique drugs in testing: {np.unique(drug_index_Te)}")
    # Convert to torch tensors
    drug_index_Tr = torch.from_numpy(drug_index_Tr).long().to(device)
    drug_index_Te = torch.from_numpy(drug_index_Te).long().to(device)
    num_drug_per_point = drug_index_Tr.shape[1]
    args.num_drug_per_point = num_drug_per_point

    # Argument for drug_encoder
    args.drug_input_dim = 5
    args.train_drug_hidden_dim = 10
    args.train_drug_output_dim = 10
    args.train_drug_encoder_layers = 3
    args.train_drug_encoder_dropout = 0.2
    args.train_drug_bn = True
    args.train_drug_layer_type = "gat"
    args.train_drug_encoder_activation = "relu"

    # Build model
    model = build_model(args, device)

    # Train the model
    train(args, pretrain_model, model, device, xTr, xTe, yTr, yTe, all_edge_index, internal_edge_index, ppi_edge_index, 
          drug_index_Tr, drug_index_Te, x_name_emb, x_desc_emb, x_bio_emb, config_groups)

# python train_res.py --train_lr 0.0005 --train_batch_size 3 --train_base_layer gat --downstream_task cell_type --label_column cell_type --tissue_general brain --disease_name "Alzheimer's Disease" --sample_ratio 0.1 --dataset_output_dir ./Output/data_ad_celltype_0.1 --train_test_random_seed 42