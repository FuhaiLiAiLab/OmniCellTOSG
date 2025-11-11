import os
import pandas as pd
import numpy as np
from pathlib import Path

from datetime import datetime

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
from CellTOSG_Downstream.embed import CellTOSG_Class, DownGNNEncoder
from CellTOSG_Foundation.utils import tab_printer
from CellTOSG_Foundation.model import CellTOSG_Foundation, DegreeDecoder, EdgeDecoder, GNNEncoder
from CellTOSG_Foundation.mask import MaskEdge
from CellTOSG_Foundation.lm_model import TextEncoder, RNAGPT_LM, ProtGPT_LM

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
    pretrain_model = CellTOSG_Foundation(text_input_dim=args.pre_lm_emb_dim,
                    omic_input_dim=args.num_omic_feature,
                    cross_fusion_output_dim=args.pre_cross_fusion_output_dim, 
                    text_encoder=text_encoder,
                    rna_seq_encoder=rna_seq_encoder,
                    prot_seq_encoder=prot_seq_encoder,
                    encoder=graph_encoder,
                    internal_encoder=internal_graph_encoder,
                    edge_decoder=edge_decoder,
                    degree_decoder=degree_decoder,
                    mask=mask).to(device)
    return pretrain_model


def build_model(args, device):
    # Build the text, rna and protein sequence encoders
    text_encoder = TextEncoder(args.text_lm_model_path, device)
    rna_seq_encoder = RNAGPT_LM(args.rna_seq_lm_model_path, args.rna_model_name, device)
    prot_seq_encoder = ProtGPT_LM(args.prot_model_name, device)
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
                    text_encoder=text_encoder,
                    rna_seq_encoder=rna_seq_encoder,
                    prot_seq_encoder=prot_seq_encoder,
                    encoder=graph_encoder,
                    internal_encoder=internal_graph_encoder).to(device)
    return model

def embed_model(dataset_loader, current_cell_num, num_entity, name_embeddings, desc_embeddings, seq_embeddings, pretrain_model, model, device, args):
    batch_loss = 0
    for batch_idx, data in enumerate(dataset_loader):
        x = Variable(data.x.float(), requires_grad=False).to(device)
        internal_edge_index = Variable(data.internal_edge_index, requires_grad=False).to(device)
        ppi_edge_index = Variable(data.edge_index, requires_grad=False).to(device)
        edge_index = Variable(data.all_edge_index, requires_grad=False).to(device)
        label = Variable(data.label, requires_grad=False).to(device)
        # Use pretrained model to get the embedding
        z = pretrain_model.internal_encoder(x, internal_edge_index) + x
        pre_x = pretrain_model.encoder.get_embedding(z, ppi_edge_index, mode='last') # mode='cat'
        output, ypred, embed_out = model(x, pre_x, edge_index, internal_edge_index, ppi_edge_index, num_entity, name_embeddings, desc_embeddings, seq_embeddings, current_cell_num)
        loss = model.loss(output, label)
        batch_loss += loss.item()
        print('Label: ', label)
        print('Prediction: ', ypred)
        batch_acc = accuracy_score(label.cpu().numpy(), ypred.cpu().numpy())
        batch_pre_embed = pre_x.view(current_cell_num, num_entity, -1).detach().cpu().numpy()
        batch_embed = embed_out.detach().cpu().numpy()
    return model, batch_loss, batch_acc, batch_pre_embed, batch_embed


def embed(args, pretrain_model, model, xAll, yAll, all_edge_index, internal_edge_index, ppi_edge_index, x_name_emb, x_desc_emb, x_bio_emb, device):
    print('-------------------------- EMBED START --------------------------')
    print('-------------------------- EMBED START --------------------------')
    print('-------------------------- EMBED START --------------------------')
    print('-------------------------- EMBED START --------------------------')
    print('-------------------------- EMBED START --------------------------')
    
    print('--- LOADING ALL FILES ... ---')
    print('xAll: ', xAll.shape)
    print('yAll: ', yAll.shape)
    analysis_num_cell = xAll.shape[0]
    num_entity = xAll.shape[1]
    num_feature = args.num_omic_feature
    analysis_batch_size = 4
    
    # Run analysis model
    model.eval()
    pre_embed_tensor = np.zeros((1, num_entity, args.pre_graph_output_dim))
    embed_tensor = np.zeros((1, num_entity))
    upper_index = 0
    batch_loss_list = []
    for index in range(0, analysis_num_cell, analysis_batch_size):
        if (index + analysis_batch_size) < analysis_num_cell:
            upper_index = index + analysis_batch_size
        else:
            upper_index = analysis_num_cell
        geo_datalist = read_batch(index, upper_index, xAll, yAll, num_feature, num_entity, all_edge_index, internal_edge_index, ppi_edge_index)
        dataset_loader = GeoGraphLoader.load_graph(geo_datalist, analysis_batch_size, args.train_num_workers)
        print('EMBED MODEL...')
        current_cell_num = upper_index - index # current batch size
        model, batch_loss, batch_acc, batch_pre_embed, batch_embed = embed_model(dataset_loader, current_cell_num, num_entity, x_name_emb, x_desc_emb, x_bio_emb, pretrain_model, model, device, args)
        print('BATCH LOSS: ', batch_loss)
        batch_loss_list.append(batch_loss)
        print('BATCH ACCURACY: ', batch_acc)
        # vstack to accumulate the results
        pre_embed_tensor = np.vstack((pre_embed_tensor, batch_pre_embed))
        embed_tensor = np.vstack((embed_tensor, batch_embed))
    # Delete the first row of zeros using np.delete
    pre_embed_tensor = np.delete(pre_embed_tensor, 0, axis=0)
    embed_tensor = np.delete(embed_tensor, 0, axis=0)
    print('pre_embed_tensor: ', pre_embed_tensor.shape)
    print('embed_tensor: ', embed_tensor.shape)
    # Save the embedding results
    np.save(os.path.join(saved_model_path, 'xTe_preembed.npy'), pre_embed_tensor)
    np.save(os.path.join(saved_model_path, 'xTe_embed.npy'), embed_tensor)


if __name__ == "__main__":
    import sys
    
    downstream_tasks = ["disease", "gender", "cell_type"]
    diseases = [
        "Alzheimers_Disease",
        "Lung_Adenocarcinoma",
        "Crohn_disease",
        "Lupus_Erythematosus,_Systemic",
    ]
    
    # Check if path is provided as command line argument
    if len(sys.argv) > 1:
        saved_model_path = Path(sys.argv[1])
        print(f"Using provided model path: {saved_model_path}")
    else:
        # Default path (you can remove this if you always want to pass the path)
        print("Error: Please provide the saved_model_path as an argument")
        print("Usage: python embed.py <saved_model_path>")
        sys.exit(1)

    config_file = saved_model_path / "config.yaml"
    args, config_groups = load_and_merge_configs(str(config_file))

    def _absorb_group(ns, group_name):
        grp = getattr(ns, group_name, None)
        if isinstance(grp, dict):
            for k, v in grp.items():
                setattr(ns, k, v)

    for group in ("dataloader", "pretraining", "training"):
        _absorb_group(args, group)

    for group in ("dataloader", "pretraining", "training"):
        if hasattr(args, group):
            delattr(args, group)

    best_model_file = saved_model_path / "best_train_model.pt"

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

    # Build Pretrain Model
    pretrain_model = build_pretrain_model(args, device)
    pretrain_model.load_state_dict(torch.load(args.pretrained_model_save_path, map_location=device))
    pretrain_model.eval()
    # Load the dataset
    x_bio_emb = np.load('/storage1/fs1/fuhai.li/Active/tianqi.x/OmniCellTOSG/dataset_outputs/x_bio_emb.npy', allow_pickle=True)
    x_desc_emb = np.load('/storage1/fs1/fuhai.li/Active/tianqi.x/OmniCellTOSG/dataset_outputs/x_desc_emb.npy', allow_pickle=True)
    x_name_emb = np.load('/storage1/fs1/fuhai.li/Active/tianqi.x/OmniCellTOSG/dataset_outputs/x_name_emb.npy', allow_pickle=True)
    all_edge_index = np.load('/storage1/fs1/fuhai.li/Active/tianqi.x/OmniCellTOSG/dataset_outputs/edge_index.npy', allow_pickle=True)
    internal_edge_index = np.load('/storage1/fs1/fuhai.li/Active/tianqi.x/OmniCellTOSG/dataset_outputs/internal_edge_index.npy', allow_pickle=True)
    ppi_edge_index = np.load('/storage1/fs1/fuhai.li/Active/tianqi.x/OmniCellTOSG/dataset_outputs/ppi_edge_index.npy', allow_pickle=True)
    xTe = np.load(os.path.join(saved_model_path, 'xTe.npy'), allow_pickle=True)
    yTe = np.load(os.path.join(saved_model_path, 'yTe.npy'), allow_pickle=True)
    print('xTe: ', xTe.shape)
    print('yTe: ', yTe.shape)
    num_classes = len(np.unique(yTe))
    args.num_class = num_classes
    args.num_entity = xTe.shape[1]
    # Convert to torch tensor
    xTe =  torch.from_numpy(xTe).float().to(device)
    yTe =  torch.from_numpy(yTe).long().to(device)
    x_name_emb = torch.from_numpy(x_name_emb).float().to(device)
    x_desc_emb = torch.from_numpy(x_desc_emb).float().to(device)
    x_bio_emb = torch.from_numpy(x_bio_emb).float().to(device)
    all_edge_index = torch.from_numpy(all_edge_index).long()
    internal_edge_index = torch.from_numpy(internal_edge_index).long()
    ppi_edge_index = torch.from_numpy(ppi_edge_index).long()

    # Build model
    model = build_model(args, device)
    model.load_state_dict(torch.load(best_model_file, map_location=device))
    # Analyze the model
    embed(args, pretrain_model, model, xTe, yTe, all_edge_index, internal_edge_index, ppi_edge_index, x_name_emb, x_desc_emb, x_bio_emb, device)