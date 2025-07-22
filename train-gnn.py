import os
import argparse
import pandas as pd
import numpy as np

from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from tqdm.auto import tqdm

from torch.autograd import Variable
from torch.utils.data import DataLoader

# custom dataloader
from GeoDataLoader.read_geograph import read_batch
from GeoDataLoader.geograph_sampler import GeoGraphLoader


from GraphModel.model import GraphDecoder
from GraphModel.utils import tab_printer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

# custom dataloader
from GeoDataLoader.read_geograph import read_batch
from GeoDataLoader.geograph_sampler import GeoGraphLoader

# custom modules
from CellTOSG_Foundation.utils import tab_printer
from CellTOSG_Loader import CellTOSGDataLoader

# Config loading
from utils import load_and_merge_configs, save_updated_config


def build_model(args, device):
    model = GraphDecoder(model_name=args.bl_train_model_name,
                        input_dim=args.bl_train_train_input_dim, 
                        hidden_dim=args.bl_train_train_hidden_dim, 
                        embedding_dim=args.bl_train_train_embedding_dim, 
                        num_node=args.num_entity, 
                        num_head=args.bl_train_num_head, 
                        device=device, 
                        num_class=args.num_class).to(device)
    return model


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
        print(f"Class {class_idx} : {class_count} samples ({percentage:.1f}%)")

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


def train_model(train_dataset_loader, current_cell_num, model, device, args, learning_rate):
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=learning_rate, eps=1e-7, weight_decay=1e-20)
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
        output, ypred = model(x, edge_index, internal_edge_index, ppi_edge_index, current_cell_num)
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


def test_model(test_dataset_loader, current_cell_num, model, device, args):
    batch_loss = 0
    all_ypred = np.zeros((1, 1))
    for batch_idx, data in enumerate(test_dataset_loader):
        x = Variable(data.x.float(), requires_grad=False).to(device)
        internal_edge_index = Variable(data.internal_edge_index, requires_grad=False).to(device)
        ppi_edge_index = Variable(data.edge_index, requires_grad=False).to(device)
        edge_index = Variable(data.all_edge_index, requires_grad=False).to(device)
        label = Variable(data.label, requires_grad=False).to(device)

        output, ypred = model(x, edge_index, internal_edge_index, ppi_edge_index, current_cell_num)
        loss = model.loss(output, label)
        batch_loss += loss.item()
        print('Label: ', label)
        print('Prediction: ', ypred)
        batch_acc = accuracy_score(label.cpu().numpy(), ypred.cpu().numpy())
        all_ypred = np.vstack((all_ypred, ypred.cpu().numpy().reshape(-1, 1)))
        all_ypred = np.delete(all_ypred, 0, axis=0)
    return model, batch_loss, batch_acc, all_ypred


def train(args, device, model, xTr, xTe, yTr, yTe, all_edge_index, internal_edge_index, ppi_edge_index, config_groups):
    train_num_cell = xTr.shape[0]
    num_entity = xTr.shape[1]
    epoch_num = args.num_train_epoch
    learning_rate = args.train_lr
    train_batch_size = args.train_batch_size

    epoch_loss_list = []
    epoch_acc_list = []
    test_loss_list = []
    test_acc_list = []
    max_test_acc = 0
    max_test_acc_id = 0

    # Clean result previous epoch_i_pred files
    print(args.bl_train_model_name)
    folder_name = 'epoch_' + str(epoch_num)

    # Add timestamp to folder name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    base_path = os.path.join(
        './' + args.bl_train_result_folder,
        args.downstream_task,
        args.disease_name,
        args.bl_train_model_name
    )

    path = os.path.join(base_path, f"{folder_name}_{timestamp}")

    os.makedirs(path, exist_ok=False)

    # Save final configuration for reference
    config_save_path = os.path.join(path, 'config.yaml')
    save_updated_config(config_groups, config_save_path)
    print(f"[Config] Saved to {config_save_path}")

    for i in range(1, epoch_num + 1):
        for _ in range(5):
            print('-------------------------- EPOCH ' + str(i) + ' START --------------------------')
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
            model, batch_loss, batch_acc, batch_ypred = train_model(train_dataset_loader, current_cell_num, model, device, args, learning_rate)
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

        conf_matrix = confusion_matrix(tmp_training_input_df['label'], tmp_training_input_df['prediction'])
        print('EPOCH ' + str(i) + ' TRAINING ACCURACY: ', accuracy)
        print('EPOCH ' + str(i) + ' TRAINING CONFUSION MATRIX: ', conf_matrix)

        print('\n-------------EPOCH TRAINING ACCURACY LIST: -------------')
        print(epoch_acc_list)
        print('\n-------------EPOCH TRAINING LOSS LIST: -------------')
        print(epoch_loss_list)

        # # # Test model on test dataset
        test_acc, test_loss, tmp_test_input_df = test(args, model, xTe, yTe, all_edge_index, internal_edge_index, ppi_edge_index, device, i)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        tmp_test_input_df.to_csv(path + '/TestPred' + str(i) + '.txt', index=False, header=True)
        print('\n-------------EPOCH TEST ACCURACY LIST: -------------')
        print(test_acc_list)
        print('\n-------------EPOCH TEST MSE LOSS LIST: -------------')
        print(test_loss_list)

        if args.use_wandb:
            wandb.log({
                "epoch": i,
                "train_loss": epoch_loss_list[-1],
                "train_acc": epoch_acc_list[-1],
                "test_loss": test_loss_list[-1],
                "test_acc": test_acc_list[-1]
            })

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


def test(args, model, xTe, yTe, all_edge_index, internal_edge_index, ppi_edge_index, device, i):
    for _ in range(5):
        print('-------------------------- TEST START --------------------------')

    print('--- LOADING TESTING FILES ... ---')
    print('xTe: ', xTe.shape)
    print('yTe: ', yTe.shape)
    test_num_cell = xTe.shape[0]
    num_entity = xTe.shape[1]
    num_feature = args.num_omic_feature

    # Build Pretrain Model
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
        model, batch_loss, batch_acc, batch_ypred = test_model(test_dataset_loader, current_cell_num, model, device, args)
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
        'Configs/bl_training.yaml'
    )

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
    required_files = ["expression_matrix.npy", "labels.npy"]

    def all_required_files_exist(path, filenames):
        return all((path / f).exists() for f in filenames)

    if args.use_extracted_data and all_required_files_exist(data_dir, required_files):
        print("[Info] Using extracted data from:", data_dir)


        class FixedDataset:
            def __init__(self, dataset_root, dataset_output_dir):
                dataset_output_dir = Path(dataset_output_dir)
                dataset_root = Path(dataset_root)

                self.data = np.load(dataset_output_dir / "expression_matrix.npy")
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
            project="bl-train-celltosg",
            name=f"{args.downstream_task}_{args.disease_name}_{args.bl_train_model_name}_bs{args.train_batch_size}_lr{args.train_lr}",
            config=vars(args)
        )

    # Graph feature
    xAll = dataset.data
    yAll = dataset.labels
    all_edge_index = dataset.edge_index
    internal_edge_index = dataset.internal_edge_index
    ppi_edge_index = dataset.ppi_edge_index
    # load embeddings into torch tensor
    xAll = torch.from_numpy(xAll).float().to(device)
    yAll = torch.from_numpy(yAll).long().to(device)
    all_edge_index = torch.from_numpy(all_edge_index).long()
    internal_edge_index = torch.from_numpy(internal_edge_index).long()
    ppi_edge_index = torch.from_numpy(ppi_edge_index).long()
    # Split dataset into train and test for xAll and yAll
    xTr, xTe, yTr, yTe, num_classes = split_dataset(xAll, yAll, args)
    args.num_class = num_classes
    args.num_entity = xAll.shape[1]

    # Build Pretrain Model
    num_feature = args.num_omic_feature
    args.num_class = num_classes
    args.num_entity = xAll.shape[1]
    # Train the model depends on the task
    model = build_model(args, device)
    # Train the model
    train(args, device, model, xTr, xTe, yTr, yTe, all_edge_index, internal_edge_index, ppi_edge_index, config_groups)