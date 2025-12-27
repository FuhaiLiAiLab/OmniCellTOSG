import os
import torch
import numpy as np
import argparse
import sys
from typing import Dict, Any, Tuple
import ast

from pynvml import *
import yaml

# GET [gpu_ids] OF ALL AVAILABLE GPUs
def get_available_devices():
    if torch.cuda.is_available():
        # GET AVAILABLE GPUs RESOURCES
        gpu_ids = []
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        # RANK THOSE GPUs
        nvmlInit()
        gpu_free_dict = {}
        for gpu_id in gpu_ids:
            h = nvmlDeviceGetHandleByIndex(gpu_id)
            info = nvmlDeviceGetMemoryInfo(h)
            gpu_free_dict[gpu_id] = info.free
            print(f'--- GPU{gpu_id} has {info.free/1024**3}G Free Memory ---')
            # import pdb; pdb.set_trace()
        sort_gpu_free = sorted(gpu_free_dict.items(), key=lambda x: x[1], reverse=True)
        sort_gpu_ids = []
        sort_gpu_ids += [gpu_id[0] for gpu_id in sort_gpu_free]
        # USE THE MAXIMAL GPU AS MAIN DEVICE
        max_free_gpu_id = sort_gpu_ids[0]
        device = torch.device(f'cuda:{gpu_ids[max_free_gpu_id]}') 
        torch.cuda.set_device(device)
        return device, sort_gpu_ids
    else:
        gpu_ids = []
        device = torch.device('cpu')
        return device, gpu_ids

def parse_int_list(v):
    if isinstance(v, list):
        return [int(x) for x in v]

    if isinstance(v, str):
        s = v.strip()
        obj = ast.literal_eval(s)
        if not isinstance(obj, list):
            raise argparse.ArgumentTypeError("must be a list, e.g. [32,32]")
        return [int(x) for x in obj]

    raise argparse.ArgumentTypeError("must be a list or a string like '[32,32]'")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file with automatic type conversion."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Get type information from the argument parser
    parser = argparse.ArgumentParser()
    
    # Recreate the same arguments as in parse_command_line_overrides() to get type info
    type_map = {}
    
    # Dataset parameters
    type_map.update({
        'data_root': str, 'extract_mode': str, 'tissue_general': str, 'tissue': str,
        'suspension_type': str, 'cell_type': str, 'disease_name': str, 'gender': str,
        'task': str, 'label_column': str, 'dataset_output_dir': str,
        'shuffle': bool, 'stratified_balancing': bool, 'train_text': bool, 'train_bio': bool,
        'sample_ratio': float, 'sample_size': int, 'random_state': int, 'correction_method': str
    })
    
    # Pretraining parameters
    type_map.update({
        'pretrain_batch_size': int, 'pre_alpha': float, 'pretrain_num_workers': int,
        'pre_lr': float, 'pre_weight_decay': float, 'p': float,
        'pretrain_text_batch_size': int, 'text_lm_model_path': str,
        'rna_seq_lm_model_path': str, 'rna_model_name': str, 'rna_seq_max_len': int,
        'prot_model_name': str
    })

    # Pretraining base layer
    type_map.update({
        'pretrain_base_layer': str
    })
    
    # Pretraining internal GNN parameters
    type_map.update({
        'num_omic_feature': int, 'pre_internal_input_dim': int,
        'pre_internal_output_dim': int, 'pre_internal_encoder_layers': int,
        'pre_internal_encoder_dropout': float, 'pre_internal_bn': bool,
        'pre_internal_layer_type': str, 'pre_internal_encoder_activation': str
    })
    
    # Pretraining graph GNN parameters
    type_map.update({
        'pre_graph_input_dim': int, 'pre_graph_output_dim': int,
        'pre_graph_encoder_layers': int, 'pre_graph_encoder_dropout': float,
        'pre_graph_bn': bool, 'pre_graph_layer_type': str,
        'pre_graph_encoder_activation': str
    })
    
    # Pretraining decoder parameters
    type_map.update({
        'pre_decoder_dim': int, 'pre_decoder_layers': int,
        'pre_decoder_dropout': float, 'pre_lm_emb_dim': int,
        'pre_cross_fusion_output_dim': int, 'entity_mlp_dims': parse_int_list, 'pretrained_model_save_path': str
    })

    # Baseline parameters
    type_map.update({
        'bl_train_train_input_dim': int, 'bl_train_train_hidden_dim': int,
        'bl_train_train_embedding_dim': int, 'bl_train_num_head': int,
        'bl_train_result_folder': str, 'bl_train_model_name': str
    })
    
    # Training parameters
    type_map.update({
        'train_test_split_ratio': float, 'train_test_random_seed': int,
        'train_lr': float, 'train_eps': float, 'train_weight_decay': float,
        'num_train_epoch': int, 'train_batch_size': int, 'train_num_workers': int
    })

    # Training base layer parameters
    type_map.update({
        'train_base_layer': str
    })

    # Training internal GNN parameters
    type_map.update({
        'train_internal_input_dim': int, 'train_internal_hidden_dim': int, 'train_internal_output_dim': int,
        'train_internal_encoder_layers': int, 'train_internal_encoder_dropout': float,
        'train_internal_bn': bool, 'train_internal_layer_type': str,
        'train_internal_encoder_activation': str
    })
    
    # Training graph GNN parameters
    type_map.update({
        'train_graph_input_dim': int, 'train_graph_hidden_dim': int, 'train_graph_output_dim': int,
        'train_graph_encoder_layers': int, 'train_graph_encoder_dropout': float,
        'train_graph_bn': bool, 'train_graph_layer_type': str,
        'train_graph_encoder_activation': str
    })
    
    # Training fusion parameters
    type_map.update({
        'train_lm_emb_dim': int, 'train_cross_fusion_output_dim': int,
        'pre_input_output_dim': int, 'final_fusion_output_dim': int,
        'train_linear_activation': str, 'train_linear_dropout': float,
        'train_result_folder': str, 'model_name': str
    })
    
    # Convert config values to correct types
    for key, value in config.items():
        if key in type_map and value is not None:
            target_type = type_map[key]
            try:
                if target_type == bool:
                    # Handle boolean conversion
                    if isinstance(value, str):
                        config[key] = value.lower() in ('true', 'yes', '1', 'on')
                    else:
                        config[key] = bool(value)
                elif target_type in (int, float):
                    # Handle numeric conversion
                    config[key] = target_type(value)
                elif target_type == str:
                    # Ensure string type
                    config[key] = str(value)
                else:
                    # Handle custom types like parse_int_list
                    config[key] = target_type(value)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert {key}='{value}' to {target_type.__name__}: {e}")
    
    return config

def parse_command_line_overrides() -> Dict[str, Any]:
    """Parse command line arguments and return as dictionary for config override."""
    parser = argparse.ArgumentParser(description='Override YAML config parameters')
    
    # ================================ DATA LOADING PARAMETERS ================================
    # Dataset parameters
    parser.add_argument('--data_root', type=str, help='Root directory for dataset')
    parser.add_argument('--categories', type=str, help='Dataset categories')
    parser.add_argument('--tissue_general', type=str, help='General tissue type')
    parser.add_argument('--tissue', type=str, help='Specific tissue type')
    parser.add_argument('--suspension_type', type=str, help='Suspension type')
    parser.add_argument('--cell_type', type=str, help='Cell type')
    parser.add_argument('--disease_name', type=str, help='Disease name')
    parser.add_argument('--gender', type=str, help='Gender')
    parser.add_argument('--downstream_task', type=str, help='Downstream task type')
    parser.add_argument('--label_column', type=str, help='Label column name')
    parser.add_argument('--shuffle', type=lambda x: x.lower() == 'true', help='Whether to shuffle dataset')
    parser.add_argument('--sample_ratio', type=float, help='Sample ratio')
    parser.add_argument('--sample_size', type=int, help='Sample size')
    parser.add_argument('--balanced', type=lambda x: x.lower() == 'true', help='Whether to balance dataset')
    parser.add_argument('--random_state', type=int, help='Random state')
    parser.add_argument('--train_text', type=lambda x: x.lower() == 'true', help='Whether to train text embeddings')
    parser.add_argument('--train_bio', type=lambda x: x.lower() == 'true', help='Whether to train bio embeddings')
    parser.add_argument('--dataset_output_dir', type=str, help='Dataset output directory')
    
    # ================================ PRETRAINING PARAMETERS ================================
    # Pretraining hyperparameters
    parser.add_argument('--pretrain_batch_size', type=int, help='Pretraining batch size')
    parser.add_argument('--pre_alpha', type=float, help='Pretraining alpha parameter')
    parser.add_argument('--pretrain_num_workers', type=int, help='Number of workers for pretraining')
    parser.add_argument('--pre_lr', type=float, help='Pretraining learning rate')
    parser.add_argument('--pre_weight_decay', type=float, help='Pretraining weight decay')
    
    # Pretraining base layer
    parser.add_argument('--pretrain_base_layer', type=str, help='Base layer type for pretraining (e.g., gcn, gat, gin, transformer)')

    # Pretraining masking parameters
    parser.add_argument('--p', type=float, help='Masking probability')
    
    # Pretraining text/rna/prot encoder parameters
    parser.add_argument('--pretrain_text_batch_size', type=int, help='Text pretraining batch size')
    parser.add_argument('--text_lm_model_path', type=str, help='Text language model path')
    parser.add_argument('--rna_seq_lm_model_path', type=str, help='RNA sequence language model path')
    parser.add_argument('--rna_model_name', type=str, help='RNA model name')
    parser.add_argument('--rna_seq_max_len', type=int, help='RNA sequence max length')
    parser.add_argument('--prot_model_name', type=str, help='Protein model name')
    
    # Pretraining internal GNN encoder parameters
    parser.add_argument('--num_omic_feature', type=int, help='Number of omic features')
    parser.add_argument('--pre_internal_input_dim', type=int, help='Pretraining internal input dimension')
    parser.add_argument('--pre_internal_output_dim', type=int, help='Pretraining internal output dimension')
    parser.add_argument('--pre_internal_encoder_layers', type=int, help='Pretraining internal encoder layers')
    parser.add_argument('--pre_internal_encoder_dropout', type=float, help='Pretraining internal encoder dropout')
    parser.add_argument('--pre_internal_bn', type=lambda x: x.lower() == 'true', help='Pretraining internal batch norm')
    parser.add_argument('--pre_internal_layer_type', type=str, help='Pretraining internal layer type')
    parser.add_argument('--pre_internal_encoder_activation', type=str, help='Pretraining internal encoder activation')
    
    # Pretraining graph GNN encoder parameters
    parser.add_argument('--pre_graph_input_dim', type=int, help='Pretraining graph input dimension')
    parser.add_argument('--pre_graph_output_dim', type=int, help='Pretraining graph output dimension')
    parser.add_argument('--pre_graph_encoder_layers', type=int, help='Pretraining graph encoder layers')
    parser.add_argument('--pre_graph_encoder_dropout', type=float, help='Pretraining graph encoder dropout')
    parser.add_argument('--pre_graph_bn', type=lambda x: x.lower() == 'true', help='Pretraining graph batch norm')
    parser.add_argument('--pre_graph_layer_type', type=str, help='Pretraining graph layer type')
    parser.add_argument('--pre_graph_encoder_activation', type=str, help='Pretraining graph encoder activation')
    
    # Pretraining edge and degree decoder parameters
    parser.add_argument('--pre_decoder_dim', type=int, help='Pretraining decoder dimension')
    parser.add_argument('--pre_decoder_layers', type=int, help='Pretraining decoder layers')
    parser.add_argument('--pre_decoder_dropout', type=float, help='Pretraining decoder dropout')
    
    # Pretraining cross fusion parameters
    parser.add_argument('--pre_lm_emb_dim', type=int, help='Pretraining language model embedding dimension')
    parser.add_argument('--pre_cross_fusion_output_dim', type=int, help='Pretraining cross fusion output dimension')

    # Entity mlp dimensions
    parser.add_argument('--entity_mlp_dims', type=parse_int_list, help='Entity MLP hidden dimensions, e.g. [32,32]')
    
    # Pretraining model save path
    parser.add_argument('--pretrained_model_save_path', type=str, help='Pretrained model save path')

    # ================================ BASELINE PARAMETERS ================================
    # Baseline-specific architecture settings
    parser.add_argument('--bl_train_train_input_dim', type=int, help='Baseline model input dimension')
    parser.add_argument('--bl_train_train_hidden_dim', type=int, help='Baseline model hidden dimension')
    parser.add_argument('--bl_train_train_embedding_dim', type=int, help='Baseline model embedding dimension')
    parser.add_argument('--bl_train_num_head', type=int, help='Number of attention heads (for transformer-like baseline models)')

    # Baseline result saving
    parser.add_argument('--bl_train_result_folder', type=str, help='Folder to save baseline training results')
    parser.add_argument('--bl_train_model_name', type=str, help='Baseline model name (e.g., gcn, gat, gin, transformer)')


    # ================================ TRAINING PARAMETERS ================================
    # Train-test dataset split parameters
    parser.add_argument('--train_test_split_ratio', type=float, help='Train-test split ratio')
    parser.add_argument('--train_test_random_seed', type=int, help='Train-test random seed')
    
    # Downstream task model hyperparameters
    parser.add_argument('--train_lr', type=float, help='Training learning rate')
    parser.add_argument('--train_eps', type=float, help='Training epsilon')
    parser.add_argument('--train_weight_decay', type=float, help='Training weight decay')
    parser.add_argument('--num_train_epoch', type=int, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, help='Training batch size')
    parser.add_argument('--train_num_workers', type=int, help='Number of workers for training')

    # Training base layer
    parser.add_argument('--train_base_layer', type=str, help='Base layer type for training (e.g., gcn, gat, gin, transformer)')

    # Training internal GNN encoder parameters
    parser.add_argument('--train_internal_input_dim', type=int, help='Training internal input dimension')
    parser.add_argument('--train_internal_hidden_dim', type=int, help='Training internal hidden dimension')
    parser.add_argument('--train_internal_output_dim', type=int, help='Training internal output dimension')
    parser.add_argument('--train_internal_encoder_layers', type=int, help='Training internal encoder layers')
    parser.add_argument('--train_internal_encoder_dropout', type=float, help='Training internal encoder dropout')
    parser.add_argument('--train_internal_bn', type=lambda x: x.lower() == 'true', help='Training internal batch norm')
    parser.add_argument('--train_internal_layer_type', type=str, help='Training internal layer type')
    parser.add_argument('--train_internal_encoder_activation', type=str, help='Training internal encoder activation')
    
    # Training graph GNN encoder parameters
    parser.add_argument('--train_graph_input_dim', type=int, help='Training graph input dimension')
    parser.add_argument('--train_graph_hidden_dim', type=int, help='Training graph hidden dimension')
    parser.add_argument('--train_graph_output_dim', type=int, help='Training graph output dimension')
    parser.add_argument('--train_graph_encoder_layers', type=int, help='Training graph encoder layers')
    parser.add_argument('--train_graph_encoder_dropout', type=float, help='Training graph encoder dropout')
    parser.add_argument('--train_graph_bn', type=lambda x: x.lower() == 'true', help='Training graph batch norm')
    parser.add_argument('--train_graph_layer_type', type=str, help='Training graph layer type')
    parser.add_argument('--train_graph_encoder_activation', type=str, help='Training graph encoder activation')
    
    # Training modal fusion parameters
    parser.add_argument('--train_lm_emb_dim', type=int, help='Training language model embedding dimension')
    parser.add_argument('--train_cross_fusion_output_dim', type=int, help='Training cross fusion output dimension')
    parser.add_argument('--pre_input_output_dim', type=int, help='Pretraining input output dimension')
    parser.add_argument('--final_fusion_output_dim', type=int, help='Final fusion output dimension')
    parser.add_argument('--train_linear_hidden_dims', type=int, nargs='+', help='Training linear hidden dimensions (space-separated list)')
    parser.add_argument('--train_linear_activation', type=str, help='Training linear activation')
    parser.add_argument('--train_linear_dropout', type=float, help='Training linear dropout')
    
    # Training result saving names
    parser.add_argument('--train_result_folder', type=str, help='Training result folder')
    parser.add_argument('--model_name', type=str, help='Model name')
   
    # ================================ GENERAL PARAMETERS ================================
    # Device parameter
    parser.add_argument('--device', type=int, help='Device ID for GPU')
    
    # Parse only known arguments to avoid conflicts
    args, unknown = parser.parse_known_args()
    
    # Convert to dictionary and filter out None values
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    
    return overrides

def merge_configs(*configs, overrides=None):
    merged = {}
    for config in configs:
        merged.update(config)
    if overrides:
        merged.update(overrides)
    return argparse.Namespace(**merged)

def load_and_merge_configs(*config_paths: str) -> Tuple[argparse.Namespace, dict]:
    """
    Load multiple YAML configs, merge them (with CLI override), and
    return both a Namespace and a structured config grouped by file name.
    """
    config_groups = {}
    raw_configs = []

    for path in config_paths:
        config = load_config(path)
        filename_key = os.path.splitext(os.path.basename(path))[0]
        config_groups[filename_key] = config
        raw_configs.append(config)

    overrides = parse_command_line_overrides()
    
    if overrides:
        print("Command line overrides detected:")
        for key, value in overrides.items():
            print(f"  {key}: {value}")
        print()
        apply_overrides_to_config_groups(config_groups, overrides)

    merged_config = merge_configs(*raw_configs, overrides)
    return merged_config, config_groups

def save_updated_config(config_groups: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config_groups, f, default_flow_style=False, sort_keys=False)
    print(f"Structured config saved to: {output_path}")

def apply_overrides_to_config_groups(config_groups: Dict[str, dict], overrides: Dict[str, Any]):
    """
    Apply flat key-value overrides to nested config_groups.
    Attempts to intelligently match keys to the correct config group.
    """
    for key, value in overrides.items():
        matched = False
        for group_name, group in config_groups.items():
            if key in group:
                group[key] = value
                matched = True
                break
        if not matched:
            # If not found in any group, default to the first one
            first_group = list(config_groups.keys())[0]
            config_groups[first_group][key] = value

    training_cfg = config_groups.get("training", {})
    base_layer = training_cfg.get("train_base_layer")
    if base_layer:
        training_cfg["train_internal_layer_type"] = base_layer
        training_cfg["train_graph_layer_type"] = base_layer
    
    pretrain_cfg = config_groups.get("pretraining", {})
    pretrain_path = pretrain_cfg.get("pretrained_model_save_path")
    pretrain_layer = pretrain_cfg.get("pretrain_base_layer")  # From training config
    if pretrain_path and "{pretrain_base_layer}" in pretrain_path and pretrain_layer:
        # Format the pretrain path with the base layer type
        pretrain_cfg["pretrained_model_save_path"] = pretrain_path.format(pretrain_base_layer=pretrain_layer)