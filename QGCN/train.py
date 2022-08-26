import json
import os
import time

from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from zmq import device

from data_process.dataset import MoleculeDataset  
from model.model import *  
from utils import get_score


class Task():
    def __init__(self, model, dataset, train_idx, valid_idx):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(),lr=0.001)
        self.criterion=nn.MSELoss()
        
        train_subsampler = SubsetRandomSampler(train_idx)
        valid_subsampler = SubsetRandomSampler(valid_idx)

        self.train_loader = DataLoader(dataset, shuffle=False, batch_size=128, sampler=train_subsampler)
        self.valid_loader = DataLoader(dataset, shuffle=False, batch_size=128, sampler=valid_subsampler)
    
    def train(self):
        self.model.train()
        loss_per_epoch_train = 0
        label_lst = []
        train_pred = []
        for data in self.train_loader:
            node_feature = data.x.to(torch.float32).to(device)
            # node_feature = data.x.to(device)
            # print(node_feature.dtype)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(torch.int32).to(device)
            sequence_input = data.sequence.to(device)
            batch = data.batch.to(device)
            label = data.y.to(torch.float32).to(device)
            
            self.optimizer.zero_grad(set_to_none=True)
            # predict = self.model(node_feature, edge_index, edge_attr, sequence_input, batch)
            predict = self.model(node_feature, edge_index, sequence_input, batch)
            label_lst.append(label)
            train_pred.append(predict)
            loss = self.criterion(predict, label)
            loss.backward()
            self.optimizer.step()         
            loss_per_epoch_train += loss.item()
            
        loss_per_epoch_train = loss_per_epoch_train / len(self.train_loader)
        return loss_per_epoch_train, torch.cat(train_pred, dim=0).tolist(), torch.cat(label_lst, dim=0).tolist()
    
    @torch.no_grad()
    def valid(self):
        loss_per_epoch_test = 0
        self.model.eval()
        label_lst = []
        test_pred = []
        for data in self.valid_loader:
            # node_feature = data.x.to(device)
            node_feature = data.x.to(torch.float32).to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(torch.int32).to(device)
            sequence_input = data.sequence.to(device)
            batch = data.batch.to(device)
            label = data.y.to(torch.float32).to(device)
            # predict = self.model(node_feature, edge_index, edge_attr, sequence_input, batch)
            predict = self.model(node_feature, edge_index, sequence_input, batch)
            label_lst.append(label)
            test_pred.append(predict)
            loss = self.criterion(predict, label)
            loss_per_epoch_test += loss.item()
        
        loss_per_epoch_test = loss_per_epoch_test / len(self.valid_loader)
        return loss_per_epoch_test, torch.cat(test_pred, dim=0).tolist(), torch.cat(label_lst, dim=0).tolist()

if __name__ == "__main__":
    
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    epochs = 300
    dataset = MoleculeDataset('./data_process/train', '4PDBbind_train.csv')

    kf = KFold(n_splits=5, random_state=1024, shuffle=True)
    for kfold, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
        model = DTINet().to(device)
        task = Task(model, dataset, train_idx, valid_idx)
        train_loss_lst = []
        valid_loss_lst = []
        time_lst = []
        train_rp_lst = []
        train_rs_lst = []
        valid_rp_lst = []
        valid_rs_lst = []
        num = 0
        
        min_loss = 10.0
        start_time =time.time()
        for epoch in tqdm(range(epochs)):
            # ——————————————————————train————————————————————————
            loss_per_epoch_train, train_predict, train_label = task.train()
            execution_time = time.time() - start_time
            
            # ——————————————————————valid————————————————————————
            loss_per_epoch_valid, valid_predict, valid_label = task.valid()
            time_lst.append(execution_time)
            train_loss_lst.append(loss_per_epoch_train)
            valid_loss_lst.append(loss_per_epoch_valid)
            
            # ——————————————————————correlation——————————————————
            train_rp, train_rs, train_rmse = get_score(train_label, train_predict)
            valid_rp, valid_rs, valid_rmse = get_score(valid_label, valid_predict)
            train_rp_lst.append(train_rp)
            train_rs_lst.append(train_rs)
            valid_rp_lst.append(valid_rp)
            valid_rs_lst.append(valid_rs)
            
            # ——————————————————————save_model———————————————————
            if (loss_per_epoch_valid < min_loss) and (epoch > 200):
                min_loss = loss_per_epoch_valid
                num += 1
                if num % 2 == 0:
                    torch.save(model, f'./data_process/cache/{kfold}_1.pkl')
                else:
                    torch.save(model, f'./data_process/cache/{kfold}_2.pkl')
            # if epoch == 280:
                # torch.save(model, f'./data_process/cache/{kfold}_3.pkl')

            # ——————————————————————performance——————————————————
            print(f'kfold: {kfold} || epoch: {epoch+1}')
            print(f'train_loss: {loss_per_epoch_train:.3f} || train_rp: {train_rp:.3f} || train_rs: {train_rs:.3f} || train_rmse {train_rmse:.3f}')
            print(f'valid_loss: {loss_per_epoch_valid:.3f} || valid_rp: {valid_rp:.3f} || valid_rs: {valid_rs:.3f} || valid_rmse {valid_rmse:.3f}')

        save_path = "./data_process/data_cache/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        dict = {"train_loss": train_loss_lst, "test_loss": valid_loss_lst, "time": time_lst, "train_rp": train_rp_lst, "train_rs": train_rs_lst, "test_rp": valid_rp_lst, "test_rs": valid_rs_lst}
        with open(save_path + f"train_data{kfold}.json", "w") as f:
            json.dump(dict, f)
              
    print('Finished training ')
