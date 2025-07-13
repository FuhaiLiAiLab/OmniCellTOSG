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
from GeoDataLoader.read_geograph import read_pretrain_batch
from GeoDataLoader.geograph_sampler import GeoGraphLoader

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


def pretrain_foundation(args, device, xAll, x_name_emb, x_desc_emb, x_bio_emb, all_edge_index, internal_edge_index, ppi_edge_index):
    # Add NaN checks at the beginning
    print("Checking for NaN values in input tensors:")
    print(f"xAll has NaN: {torch.isnan(xAll).any()}")
    print(f"x_name_emb has NaN: {torch.isnan(x_name_emb).any()}")
    print(f"x_desc_emb has NaN: {torch.isnan(x_desc_emb).any()}")
    print(f"x_bio_emb has NaN: {torch.isnan(x_bio_emb).any()}")
    
    # Check for infinite values
    print(f"xAll has inf: {torch.isinf(xAll).any()}")
    print(f"x_name_emb has inf: {torch.isinf(x_name_emb).any()}")
    print(f"x_desc_emb has inf: {torch.isinf(x_desc_emb).any()}")
    print(f"x_bio_emb has inf: {torch.isinf(x_bio_emb).any()}")
    
    # Check data ranges
    print(f"xAll range: [{xAll.min().item():.6f}, {xAll.max().item():.6f}]")
    print(f"x_name_emb range: [{x_name_emb.min().item():.6f}, {x_name_emb.max().item():.6f}]")
    print(f"x_desc_emb range: [{x_desc_emb.min().item():.6f}, {x_desc_emb.max().item():.6f}]")
    print(f"x_bio_emb range: [{x_bio_emb.min().item():.6f}, {x_bio_emb.max().item():.6f}]")

    num_cell = xAll.shape[0]
    num_entity = xAll.shape[1]
    upper_index = 0
    batch_size = args.pretrain_batch_size
    num_feature = args.num_omic_feature
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
        pretrain_geo_datalist = read_pretrain_batch(index, upper_index, xAll, num_feature, num_entity, all_edge_index, internal_edge_index, ppi_edge_index)
        dataset_loader = GeoGraphLoader.load_graph(pretrain_geo_datalist, args.pretrain_batch_size, args.pretrain_num_workers) # read by batch size

        for batch_idx, data in enumerate(dataset_loader):
            print(f'Starting {index} - {upper_index}')
            print('Start Training (Link Prediction Pretext Training)...')
            optimizer = torch.optim.Adam(pretrain_model.parameters(),
                                        lr=args.lr,
                                        weight_decay=args.weight_decay)

            train_data, val_data, test_data = T.RandomLinkSplit(num_test=0.1, num_val=0.0,
                                                            is_undirected=False,
                                                            split_labels=True,
                                                            add_negative_train_samples=False)(data)
            
            pretrain_model.reset_parameters()
            train_data = train_data.to(device)
            avg_loss, step_avg_loss_list = pretrain_model.train_step(train_data,
                                        num_entity,
                                        x_name_emb, x_desc_emb, x_bio_emb,
                                        optimizer,
                                        alpha=args.alpha, 
                                        batch_size=current_cell_num)
            batch_avg_loss_list.append(avg_loss)
            all_step_avg_loss_list.extend(step_avg_loss_list)
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f'Best Loss: {best_loss}')
                torch.save(pretrain_model.state_dict(), args.pretrained_model_save_path)
            # save loss list to text file
            with open(args.pretrained_model_save_path.replace('.pt', '_batch_avg_loss_list.txt'), 'w') as f:
                for item in batch_avg_loss_list:
                    f.write("%s\n" % item)
            with open(args.pretrained_model_save_path.replace('.pt', '_all_step_avg_loss_list.txt'), 'w') as f:
                for item in all_step_avg_loss_list:
                    f.write("%s\n" % item)

            test_data = test_data.to(device)
            test_auc, test_ap = pretrain_model.test_step(test_data, 
                                    test_data.pos_edge_label_index, 
                                    test_data.neg_edge_label_index) 
            batch_auc_list.append(test_auc)
            batch_acc_list.append(test_ap)
            # save auc list to text file
            with open(args.pretrained_model_save_path.replace('.pt', '_batch_auc.txt'), 'w') as f:
                for item in batch_auc_list:
                    f.write("%s\n" % item) 
            with open(args.pretrained_model_save_path.replace('.pt', '_batch_ap.txt'), 'w') as f:
                for item in batch_acc_list:
                    f.write("%s\n" % item) 
            print(f'Link Prediction Pretraining Results:\n'
                f'AUC: {test_auc:.2%}',
                f'AP: {test_ap:.2%}')
            print(f'Pretraining {upper_index} done!')
    return pretrain_model


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
    parser.add_argument('--pretrain_batch_size', type=int, default=4, help='Batch size for pretraining. (default: 4)')
    parser.add_argument('--pretrain_text_batch_size', type=int, default=256, help='Batch size for pretraining text. (default: 256)')
    parser.add_argument('--text_lm_model_path', type=str, default='microsoft/deberta-v3-small', help='Path to the pretrained language model. (default: microsoft/deberta-v3-small)')
    parser.add_argument('--rna_seq_lm_model_path', default='./Checkpoints/pretrained_dnagpt', help='Path to the pretrained language model. (default: ./Checkpoints/pretrained_dnagpt)')
    parser.add_argument('--rna_model_name', default='dna_gpt0.1b_h', help='Name of the pretrained rna language model. (default: dna_gpt0.1b_h)')
    parser.add_argument('--prot_model_name', default='nferruz/ProtGPT2', help='Name of the pretrained protein language model. (default: nferruz/ProtGPT2)')

    parser.add_argument('--layer', type=str, default='gcn', help='GNN layer, (default: gcn)')
    parser.add_argument('--encoder_activation', type=str, default='leaky_relu', help='Activation function for GNN encoder, (default: leaky_relu)')

    parser.add_argument('--num_omic_feature', type=int, default=1, help='Omic feature size. (default: 1)')
    parser.add_argument('--lm_emb_dim', type=int, default=1, help='Text embedding dimension. (default: 1)')
    parser.add_argument('--rna_seq_max_len', type=int, default=256, help='Max length of RNA sequence. (default: 256)')

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

    parser.add_argument('--start', type=str, default='node', help='Which Type to sample starting nodes for random walks, (default: node)')
    parser.add_argument('--p', type=float, default=0.00001, help='Mask ratio for MaskEdge')

    parser.add_argument('--bn', action='store_true', help='Whether to use batch normalization for GNN encoder. (default: False)')
    parser.add_argument('--l2_normalize', action='store_true', help='Whether to use l2 normalize output embedding. (default: False)')
    parser.add_argument('--graphclas_weight_decay', type=float, default=1e-3, help='weight_decay for node classification training. (default: 1e-3)')

    parser.add_argument('--pretrained_model_save_path', type=str, default='./Checkpoints/pretrained_models/pretrained_celltosg_foundation.pt', help='save path for model. (default: pretrained_celltosg_foundation.pt)')
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
    os.makedirs(os.path.dirname(args.pretrained_model_save_path), exist_ok=True)
    pretrain_model = build_pretrain_model(args, device)

    # Prepare embeddings
    x_name_emb, x_desc_emb, x_bio_emb = pre_embed_text(args, dataset, pretrain_model, device)
    # Graph feature
    xAll = dataset.data
    all_edge_index = dataset.edge_index
    internal_edge_index = dataset.internal_edge_index
    ppi_edge_index = dataset.ppi_edge_index
    # load embeddings into torch tensor
    xAll = torch.from_numpy(xAll).float().to(device)
    x_name_emb = torch.from_numpy(x_name_emb).float().to(device)
    x_desc_emb = torch.from_numpy(x_desc_emb).float().to(device)
    x_bio_emb = torch.from_numpy(x_bio_emb).float().to(device)
    all_edge_index = torch.from_numpy(all_edge_index).long()
    internal_edge_index = torch.from_numpy(internal_edge_index).long()
    ppi_edge_index = torch.from_numpy(ppi_edge_index).long()

    # Pretrain model
    pretrain_model = pretrain_foundation(args, device, xAll, x_name_emb, x_desc_emb, x_bio_emb, all_edge_index, internal_edge_index, ppi_edge_index)