#!/bin/bash
#SBATCH --job-name=pdac_train
#SBATCH --partition=general-gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

python /storage1/fs1/fuhai.li/Active/c.weihang/OmniCellTOSG_model/train.py \
  --train_lr 0.0005 \
  --train_batch_size 3 \
  --train_base_layer gat \
  --downstream_task disease \
  --label_column disease \
  --tissue_general "pancreas" \
  --disease_name "pancreatic ductal adenocarcinoma" \
  --sample_ratio 1 \
  --dataset_output_dir ./Output/data_pdac_disease_1 \
  --train_test_random_seed 42
