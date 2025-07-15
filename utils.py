import os
import torch
import numpy as np
import argparse
import sys
from typing import Dict, Any

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

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

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
    
    # Pretraining model save path
    parser.add_argument('--pretrained_model_save_path', type=str, help='Pretrained model save path')
    
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
    
    # Training internal GNN encoder parameters
    parser.add_argument('--train_internal_input_dim', type=int, help='Training internal input dimension')
    parser.add_argument('--train_internal_output_dim', type=int, help='Training internal output dimension')
    parser.add_argument('--train_internal_encoder_layers', type=int, help='Training internal encoder layers')
    parser.add_argument('--train_internal_encoder_dropout', type=float, help='Training internal encoder dropout')
    parser.add_argument('--train_internal_bn', type=lambda x: x.lower() == 'true', help='Training internal batch norm')
    parser.add_argument('--train_internal_layer_type', type=str, help='Training internal layer type')
    parser.add_argument('--train_internal_encoder_activation', type=str, help='Training internal encoder activation')
    
    # Training graph GNN encoder parameters
    parser.add_argument('--train_graph_input_dim', type=int, help='Training graph input dimension')
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
    
    # Pretrained model path for loading
    parser.add_argument('--pretrained_model_path', type=str, help='Pretrained model path for loading')
    
    # ================================ GENERAL PARAMETERS ================================
    # Device parameter
    parser.add_argument('--device', type=int, help='Device ID for GPU')
    
    # Parse only known arguments to avoid conflicts
    args, unknown = parser.parse_known_args()
    
    # Convert to dictionary and filter out None values
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    
    return overrides

def merge_configs(*configs: Dict[str, Any]) -> argparse.Namespace:
    """Merge multiple configuration dictionaries into a single Namespace object."""
    merged_config = {}
    for config in configs:
        merged_config.update(config)
    return argparse.Namespace(**merged_config)

def load_and_merge_configs(*config_paths: str) -> argparse.Namespace:
    """Load and merge multiple YAML configuration files with command line override support."""
    # Load YAML configs
    configs = [load_config(path) for path in config_paths]
    
    # Get command line overrides
    overrides = parse_command_line_overrides()
    
    # Print override information
    if overrides:
        print("Command line overrides detected:")
        for key, value in overrides.items():
            print(f"  {key}: {value}")
        print()
    
    # Merge configs with overrides taking precedence
    merged_config = merge_configs(*configs, overrides)
    
    return merged_config

def save_updated_config(config: argparse.Namespace, output_path: str):
    """Save the final merged configuration to a YAML file."""
    config_dict = vars(config)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False, sort_keys=True)
    
    print(f"Final configuration saved to: {output_path}")