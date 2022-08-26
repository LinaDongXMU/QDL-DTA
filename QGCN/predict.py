import os
import torch
import pandas as pd
from data_process.dataset import MoleculeDataset
from torch_geometric.loader import DataLoader

train_dataset = MoleculeDataset('./data_process/train', '4PDBbind_train.csv')
train_loader = DataLoader(train_dataset, shuffle=False, batch_size=128)
test_dataset = MoleculeDataset('./data_process/test', '4PDBbind_test.csv')
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def valid(data_loader):
    model.eval()
    label_lst = []
    test_pred = []
    for data in data_loader:
        node_feature = data.x.to(torch.long).to(device)
        edge_index = data.edge_index.to(device)
        edge_attr = data.edge_attr.to(torch.int32).to(device)
        sequence_input = data.sequence.to(device)
        #pocket_input = data.pocket.to(device)
        batch = data.batch.to(device)
        label = data.y.to(torch.long).to(device)
        predict = model(node_feature, edge_index, sequence_input, batch)
        label_lst.append(label)
        test_pred.append(predict)
    
    return torch.cat(test_pred, dim=0).tolist()

df = pd.read_csv('./data_process/test/raw/4PDBbind_test.csv', encoding='utf-8')
for i, path in enumerate(os.listdir('./data_process/cache')):
    model_path = os.path.join('./data_process/cache', path)
    #print(model_path)
    model = torch.load(model_path)
    predict = valid(test_loader)
    df[f'pre{i+1}'] = predict
df.to_csv('./4PDBbind_test.csv', index=None, encoding='utf-8')

df = pd.read_csv('./data_process/train/raw/4PDBbind_train.csv', encoding='utf-8')
for i, path in enumerate(os.listdir('./data_process/cache')):
    model_path = os.path.join('./data_process/cache', path)
    model = torch.load(model_path)
    predict = valid(train_loader)
    df[f'pre{i+1}'] = predict
df.to_csv('./4PDBbind_train.csv', index=None, encoding='utf-8')
