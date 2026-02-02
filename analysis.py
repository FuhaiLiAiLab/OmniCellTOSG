import os
import pandas as pd
import numpy as np
from pathlib import Path

from datetime import datetime
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

# custom dataloader
from GeoDataLoader.read_geograph import read_batch
from GeoDataLoader.geograph_sampler import GeoGraphLoader

# custom modules
from CellTOSG_Foundation.lm_model import TextEncoder, RNAGPT_LM, ProtGPT_LM
from CellTOSG_Downstream.analyzer import CellTOSG_Class, DownGNNEncoder
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


def analyze_model(dataset_loader, current_cell_num, num_entity, name_embeddings, desc_embeddings, seq_embeddings, pretrain_model, model, device, args, index):
    batch_loss = 0
    all_ypred = np.zeros((1, 1))
    for batch_idx, data in enumerate(dataset_loader):
        x = Variable(data.x.float(), requires_grad=False).to(device)
        internal_edge_index = Variable(data.internal_edge_index, requires_grad=False).to(device)
        ppi_edge_index = Variable(data.edge_index, requires_grad=False).to(device)
        edge_index = Variable(data.all_edge_index, requires_grad=False).to(device)
        label = Variable(data.label, requires_grad=False).to(device)
        # Use pretrained model to get the embedding
        z = pretrain_model.internal_encoder(x, internal_edge_index) + x
        pre_x = pretrain_model.encoder.get_embedding(z, ppi_edge_index, mode='last') # mode='cat'
        output, ypred = model(x, pre_x, edge_index, internal_edge_index, ppi_edge_index, num_entity, name_embeddings, desc_embeddings, seq_embeddings, current_cell_num, index, args.analysis_output_dir)
        loss = model.loss(output, label)
        batch_loss += loss.item()
        print('Label: ', label)
        print('Prediction: ', ypred)
        batch_acc = accuracy_score(label.cpu().numpy(), ypred.cpu().numpy())
        all_ypred = np.vstack((all_ypred, ypred.cpu().numpy().reshape(-1, 1)))
        all_ypred = np.delete(all_ypred, 0, axis=0)
    return model, batch_loss, batch_acc, all_ypred


def analyze(args, pretrain_model, model, xAll, yAll, all_edge_index, internal_edge_index, ppi_edge_index, x_name_emb, x_desc_emb, x_bio_emb, device):
    print('-------------------------- ANALYZE START --------------------------')
    print('-------------------------- ANALYZE START --------------------------')
    print('-------------------------- ANALYZE START --------------------------')
    print('-------------------------- ANALYZE START --------------------------')
    print('-------------------------- ANALYZE START --------------------------')
    
    print('--- LOADING ALL FILES ... ---')
    print('xAll: ', xAll.shape)
    print('yAll: ', yAll.shape)
    analysis_num_cell = xAll.shape[0]
    num_entity = xAll.shape[1]
    num_feature = args.num_omic_feature
    analysis_batch_size = 4
    
    # Run analysis model
    model.eval()
    all_ypred = np.zeros((1, 1))
    upper_index = 0
    batch_loss_list = []

    total_batches = (analysis_num_cell + analysis_batch_size - 1) // analysis_batch_size

    with tqdm(total=total_batches, desc="Analyze batches", unit="batch") as pbar:
        for index in range(0, analysis_num_cell, analysis_batch_size):
            if (index + analysis_batch_size) < analysis_num_cell:
                upper_index = index + analysis_batch_size
            else:
                upper_index = analysis_num_cell
            geo_datalist = read_batch(index, upper_index, xAll, yAll, num_feature, num_entity, all_edge_index, internal_edge_index, ppi_edge_index)
            dataset_loader = GeoGraphLoader.load_graph(geo_datalist, analysis_batch_size, args.train_num_workers)
            tqdm.write("ANALYZE MODEL...")
            current_cell_num = upper_index - index # current batch size
            model, batch_loss, batch_acc, batch_ypred = analyze_model(dataset_loader, current_cell_num, num_entity, x_name_emb, x_desc_emb, x_bio_emb, pretrain_model, model, device, args, index)
            tqdm.write(f"BATCH LOSS: {batch_loss:.8f}")
            batch_loss_list.append(batch_loss)
            tqdm.write(f"BATCH ACCURACY: {batch_acc:.8f}")

            pbar.set_postfix({
            "cells": f"{upper_index}/{analysis_num_cell}",
            "loss": f"{batch_loss:.8f}",
            "acc": f"{batch_acc:.8f}",
            })
            pbar.update(1)


if __name__ == "__main__":
    # Load and merge configurations with command line override support
    saved_model_path = Path(
        "./CellTOSG_model_results/disease/Lung_Adenocarcinoma/gat/epoch_50_3_0.0005_3407_20260103_201909"
    )

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

    analysis_output_dir = Path(str(saved_model_path).replace("CellTOSG_model_results",
                                                            "CellTOSG_analysis_results"))

    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_output_dir = str(analysis_output_dir)

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
    data_dir = Path(args.dataset_output_dir)
    required_files = ["train/expression_matrix.npy", "train/labels_train.npy", "test/expression_matrix.npy", "test/labels_test.npy"]

    def all_required_files_exist(path, filenames):
        return all((path / f).exists() for f in filenames)

    if all_required_files_exist(data_dir, required_files):
        print("[Info] Using extracted data from:", data_dir)

        class FixedDataset:
            def __init__(self, dataset_root, task, dataset_output_dir):
                dataset_output_dir = Path(dataset_output_dir)
                dataset_root = Path(dataset_root)

                self.x_train = np.load(f"{dataset_output_dir}/train/expression_matrix.npy")
                self.x_test = np.load(f"{dataset_output_dir}/test/expression_matrix.npy")

                self.y_train = np.load(f"{dataset_output_dir}/train/labels_train.npy")
                self.y_test = np.load(f"{dataset_output_dir}/test/labels_test.npy")

                self.label_train = pd.read_csv(f"{dataset_output_dir}/train/labels.csv")
                self.label_test = pd.read_csv(f"{dataset_output_dir}/test/labels.csv")

                self.label_mapping_table = pd.read_csv(f"{dataset_output_dir}/label_mapping_{task}.csv")

                self.edge_index = np.load(dataset_root / "edge_index.npy")
                self.internal_edge_index = np.load(dataset_root / "internal_edge_index.npy")
                self.ppi_edge_index = np.load(dataset_root / "ppi_edge_index.npy")
                self.x_name_emb = np.load(dataset_root / "x_name_emb.npy")
                self.x_desc_emb = np.load(dataset_root / "x_desc_emb.npy")
                self.x_bio_emb = np.load(dataset_root / "x_bio_emb.npy")

        dataset = FixedDataset(args.dataset_root, args.task, args.dataset_output_dir)

        # Graph feature
        xTr = dataset.x_train
        xTe = dataset.x_test
        yTr = dataset.y_train
        yTe = dataset.y_test
        label_train = dataset.label_train
        label_test = dataset.label_test
        label_mapping_table = dataset.label_mapping_table

    else:
        raise FileNotFoundError("Required extracted data files are missing. Please check the data.")

    # Replace spaces and quotes in disease name after loading the dataset
    args.disease_name = args.disease_name.replace("'", "").replace(" ", "_")

    args.num_entity = xTr.shape[1]
    print(f"Number of entities: {args.num_entity}")

    # Build Pretrain Model
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

    print("Mapping table:")
    print(label_mapping_table)

    xAll = np.vstack((xTr, xTe))
    print(f"\n[Debug] xAll shape: {xAll.shape}")

    print("\n[Debug] xAll (start) [:5, :5]:")
    print(xAll[:5, :5])

    yTr2 = yTr.reshape(-1, 1)
    yTe2 = yTe.reshape(-1, 1)

    yAll = np.vstack((yTr2, yTe2))
    print(f"\n[Debug] yAll shape: {yAll.shape}")

    print("\n[Debug] yAll (start) [:5]:")
    print(yAll[:5])

    labelAll = pd.concat([label_train, label_test], axis=0).reset_index(drop=True)
    print("\n[Debug] labelAll (start) head(5):")
    print(labelAll.head(5))

    merged_dir = Path(str(args.analysis_output_dir.replace("CellTOSG_analysis_results", "CellTOSG_analysis_data")))
    merged_dir.mkdir(parents=True, exist_ok=True)

    xall_path = merged_dir / "xAll.npy"
    yall_path = merged_dir / "yAll.npy"
    labelall_path = merged_dir / "labelAll.csv"

    np.save(xall_path, xAll)
    np.save(yall_path, yAll)

    labelAll.to_csv(labelall_path, index=False)

    label_mapping_table.to_csv(merged_dir / f"label_mapping_{args.task}.csv", index=False)

    print(f"[Info] Saved xAll to: {xall_path}")
    print(f"[Info] Saved yAll to: {yall_path}")
    print(f"[Info] Saved labelAll to: {labelall_path}")
    print(f"[Info] Saved label_mapping_table to: {merged_dir / f'label_mapping_{args.task}.csv'}")


    # load graph into torch tensor
    xAll = torch.from_numpy(xAll).float().to(device)
    yAll = torch.from_numpy(yAll).long().view(-1, 1).to(device)

    all_edge_index = torch.from_numpy(all_edge_index).long()
    internal_edge_index = torch.from_numpy(internal_edge_index).long()
    ppi_edge_index = torch.from_numpy(ppi_edge_index).long()

    # Get unique values from yAll
    unique_values = torch.unique(yAll)
    num_classes = len(unique_values)
    args.num_class = num_classes

    # Build model
    model = build_model(args, device)
    model.load_state_dict(torch.load(best_model_file, map_location=device))
    # Analyze the model
    analyze(args, pretrain_model, model, xAll, yAll, all_edge_index, internal_edge_index, ppi_edge_index, x_name_emb, x_desc_emb, x_bio_emb, device)