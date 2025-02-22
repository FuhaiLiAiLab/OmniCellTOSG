# Omni-CellTOG

## 1. Dataset content
```
-CellTOG
    - S_name.csv (2 columns, BioMedGraphica_ID, Processed_name)
    - S_desc.csv (2 columns, BioMedGraphica_ID, Description)
    - S_bio.csv (2 columns, BioMedGraphica_ID, Sequence)
    - edge_index.npy
    - internal_edge_index.npy
    - ppi_edge_index.npy

    - (folders under organs)
    - Brain
    - Lung
    - Kidney
    - ...
```


## 2. Import Dataset

```python
import CellTOGDataset

datset_list =  CellTOGDataset(root="path", categories=["get_organ", "get_disease", "get_organ_disease"], name = "brain" / "AD" / "brain-AD", label_type = "ct" / "og" / "ds" / "status", shffule = True)
# dataset list should contain [x, y, x_name, x_desc, x_bio, edge_index, internal_edge_index (ts-pt), ppi_edge_index]

```


### 3. Running

```python
def glm(data = Dataset, graph_encoder = 'GCN' / 'GAT' / 'UniMP', text_language_encoder = 'BERT', seq_encoder = ['A', 'B', 'C'])

    return model, loss, acc, pred

```


```python
x, y, x_name, x_desc, x_bio, edge_index = dataset_list
xTr, yTr, xTe, yTe = CellTOGDataset.split(ratio=0.9)

# Bacth loading data into torch.geometric data
cell_num, node_num, feature_num = x.shape
cell_num, label_num = y.shape

batch_size = 256
for index in range(0, cell_num, batch_size):
    if (index + batch_size) < cell_num:
        upper_index = index + batch_size
    else:
        upper_index = cell_num
    geo_train_datalist = read_batch(index, upper_index, xTr, yTr, num_feature, num_node, all_edge_index, internal_edge_index, ppi_edge_index)
    train_dataset_loader = GeoGraphLoader.load_graph(geo_train_datalist, args)
    glm(loaded_dataset, )

    # model, batch_loss, batch_acc, batch_ypred = train_graphclas_model(train_dataset_loader, dna_embedding, rna_embedding, protein_embedding,
    #                                                                            model, device, args)
```