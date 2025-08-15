import os
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

# Config loading
from utils import load_and_merge_configs


def build_pretrain_model(args, device):
    # Build the mask for edge reconstruction
    mask = MaskEdge(p=float(args.p))
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
        x_name_emb = text_encoder.generate_embeddings(name_sentence_list, batch_size=args.pretrain_text_batch_size, text_emb_dim=args.pre_lm_emb_dim)
        x_desc_emb = text_encoder.generate_embeddings(desc_sentence_list, batch_size=args.pretrain_text_batch_size, text_emb_dim=args.pre_lm_emb_dim)
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

    optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=args.pre_lr, weight_decay=args.pre_weight_decay)

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

            train_data, val_data, test_data = T.RandomLinkSplit(num_test=0.1, num_val=0.0,
                                                            is_undirected=False,
                                                            split_labels=True,
                                                            add_negative_train_samples=False)(data)
            
            train_data = train_data.to(device)
            avg_loss, step_avg_loss_list = pretrain_model.train_step(train_data,
                                        num_entity,
                                        x_name_emb, x_desc_emb, x_bio_emb,
                                        optimizer,
                                        alpha=args.pre_alpha,
                                        batch_size=current_cell_num)
            batch_avg_loss_list.append(avg_loss)
            all_step_avg_loss_list.extend(step_avg_loss_list)

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

            if args.use_wandb:
                wandb.log({
                    "avg_loss": avg_loss,
                    "test_auc": test_auc,
                    "test_ap": test_ap,
                    "batch_index": batch_idx,
                    "global_index": index,
                    "upper_index": upper_index
                })

            # Save best model and metrics
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f'Best Loss: {best_loss}')
                
                # Save model
                torch.save(pretrain_model.state_dict(), args.pretrained_model_save_path)

                # Save current metrics to CSV
                best_metrics_path = args.pretrained_model_save_path.replace('.pt', '_best_metrics.csv')
                df_best = pd.DataFrame([{
                    "batch_start": index,
                    "batch_end": upper_index,
                    "avg_loss": avg_loss,
                    "auc": test_auc,
                    "ap": test_ap
                }])
                df_best.to_csv(best_metrics_path, index=False)
                print(f"[Best Model Saved] Metrics written to: {best_metrics_path}")

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


if __name__ == "__main__":
    # Load and merge configurations with command line override support
    args, config_groups = load_and_merge_configs(
        'Configs/dataloader_pretrain.yaml',
        'Configs/pretraining.yaml'
    )
    
    # Save final configuration for reference
    from utils import save_updated_config
    args.pretrained_model_save_path = args.pretrained_model_save_path.format(
    pretrain_base_layer=args.pretrain_base_layer
    )
    config_groups['pretraining']['pretrained_model_save_path'] = args.pretrained_model_save_path
    os.makedirs(os.path.dirname(args.pretrained_model_save_path), exist_ok=True)
    config_save_path = os.path.join(os.path.dirname(args.pretrained_model_save_path), 'config.yaml')
    save_updated_config(config_groups, config_save_path)
    
    print(tab_printer(args))

    args.use_wandb = False

    if args.use_wandb:
        import wandb
        wandb.init(
            project="pretrain-celltosg",
            name=f"{args.pretrain_base_layer}_lr{args.pre_lr}",
            config=vars(args)
        )

    # Check device
    if args.device < 0:
        device = 'cpu'
    else:
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    args.tissue = None
    args.suspension_type = None
    args.cell_type = None
    args.gender = None

    # # Load dataset with conditions
    # dataset = CellTOSGDataLoader(
    #     root=args.data_root,
    #     conditions={
    #         "tissue_general": args.tissue_general,
    #         # "tissue": args.tissue,
    #         # "suspension_type": args.suspension_type,
    #         # "cell_type": args.cell_type,
    #         "disease": args.disease_name,
    #         # "gender": args.gender,
    #     },
    #     downstream_task=args.downstream_task,  
    #     label_column=args.label_column,
    #     sample_ratio=args.sample_ratio,
    #     sample_size=args.sample_size,
    #     balanced=args.balanced,
    #     shuffle=args.shuffle,
    #     random_state=args.random_state,
    #     train_text=args.train_text,
    #     train_bio=args.train_bio,
    #     output_dir=args.dataset_output_dir
    # )

    class FixedDataset:
        def __init__(self, dataset_root, dataset_output_dir):
            self.data = np.load(f"{dataset_output_dir}/expression_matrix.npy")
            self.edge_index = np.load(f"{dataset_root}/edge_index.npy")
            self.internal_edge_index = np.load(f"{dataset_root}/internal_edge_index.npy")
            self.ppi_edge_index = np.load(f"{dataset_root}/ppi_edge_index.npy")
            self.x_name_emb = np.load(f"{dataset_root}/x_name_emb.npy")
            self.x_desc_emb = np.load(f"{dataset_root}/x_desc_emb.npy")
            self.x_bio_emb = np.load(f"{dataset_root}/x_bio_emb.npy")

    dataset = FixedDataset(args.dataset_root, args.dataset_output_dir)

    # Build Pretrain Model
    # os.makedirs(os.path.dirname(args.pretrained_model_save_path), exist_ok=True)
    pretrain_model = build_pretrain_model(args, device)
    pretrain_model.reset_parameters()

    # Prepare embeddings
    # x_name_emb, x_desc_emb, x_bio_emb = pre_embed_text(args, dataset, pretrain_model, device)
    x_name_emb = dataset.x_name_emb
    x_desc_emb = dataset.x_desc_emb
    x_bio_emb = dataset.x_bio_emb
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