import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from scipy.stats import pearsonr
from scipy.stats import spearmanr

class MyDataset(Dataset):
    def __init__(self, drug, target, label):
        self.drug = drug
        self.target = target
        self.label = label

    def __len__(self):
        return len(self.drug)

    def __getitem__(self, index):
        return self.drug[index], self.target[index], self.label[index]


class Task():
    def __init__(self, net, DATAFILE):
        self.net = net
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.DATAFILE = DATAFILE

    def load_data(self):

        VOCAB_LIGAND_ISO = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                            "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                            "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                            "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                            "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                            "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                            "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                            "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
        VOCAB_PROTEIN = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                         "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                         "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                         "U": 19, "T": 20, "W": 21,
                         "V": 22, "Y": 23, "X": 24,
                         "Z": 25}

        dataset = pd.read_csv(self.DATAFILE)
        # data_num = dataset.shape[0]
        drug = map(lambda x: np.pad([int(VOCAB_LIGAND_ISO[s]) for s in x], (0, 100 - len(x)))
        if len(x) < 100
        else [int(VOCAB_LIGAND_ISO[s]) for s in x][:100],
                   dataset['smiles'])
        target = map(lambda x: np.pad([int(VOCAB_PROTEIN[s]) for s in x], (0, 1200 - len(x)))
        if len(x) < 1200
        else [int(VOCAB_PROTEIN[s]) for s in x][:1200],
                     dataset['sequence'])
        drug = np.array(list(drug))
        target = np.array(list(target))
        label = np.array(dataset['label'])
        drug = torch.from_numpy(drug)
        label = torch.from_numpy(label).view(-1, 1).float()
        target = torch.from_numpy(target)
        trainset = MyDataset(drug, target, label)

        kf = KFold(n_splits=5, random_state=None, shuffle=True)
        train_loader_lst = []
        valid_loader_lst = []
        for fold, (train_idx, valid_idx) in enumerate(kf.split(trainset)):
            # print(train_idx)
            train_subsampler = SubsetRandomSampler(train_idx)
            valid_subsampler = SubsetRandomSampler(valid_idx)
            train_loader = DataLoader(trainset, batch_size=64, shuffle=False, sampler=train_subsampler)
            valid_loader = DataLoader(trainset, batch_size=64, shuffle=False, sampler=valid_subsampler)
            train_loader_lst.append(train_loader)
            valid_loader_lst.append(valid_loader)

        return train_loader_lst, valid_loader_lst

    def train(self, trainloader: torch.utils.data.DataLoader):
        """Train the cnn."""
        train_pred_lst = []
        labels_lst = []
        self.net.train()
        running_loss = 0.0

        for data in trainloader:
            drug, target, labels = data[0].type(torch.LongTensor), data[1].type(torch.LongTensor), data[2]
            drug = drug
            target = target
            labels = labels
            labels_lst.append(labels.squeeze(1).view(-1))
            self.optimizer.zero_grad()
            outputs = self.net(drug, target).squeeze(-1)
            train_pred_lst.append(outputs.squeeze(1).view(-1))
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        running_loss = running_loss / len(trainloader)
        labels_lst = torch.cat(labels_lst, dim=0).detach().cpu().numpy().reshape(-1)
        train_pred_lst = torch.cat(train_pred_lst, dim=0).detach().cpu().numpy().reshape(-1)
        train_MSE = mean_squared_error(train_pred_lst, labels_lst)
        train_Rp = pearsonr(train_pred_lst, labels_lst)
        # train_Rs = spearmanr(train_pred_lst,labels_lst)
        # train_loss = np.append(train_loss, running_loss)
        return running_loss, train_MSE, train_Rp

    @torch.no_grad()
    def test(self, test_loader):
        test_pred_lst = []
        labels_lst = []
        test_loss_all = 0
        self.net.eval()
        for i, data in enumerate(test_loader):
            drug, target, labels = data[0], data[1], data[2]
            drug = drug
            target = target
            labels = labels
            labels_lst.append(labels.squeeze(1).view(-1, 1))
            pred = self.net(drug, target).squeeze(-1)

            test_pred_lst.append(pred.squeeze(1).view(-1, 1))
            loss = self.criterion(pred, labels)
            test_loss_all += loss.item()


        test_loss = test_loss_all / (i + 1)
        labels_lst = torch.cat(labels_lst).detach().cpu().numpy().reshape(-1)
        test_pred_lst = torch.cat(test_pred_lst).detach().cpu().numpy().reshape(-1)
        test_MSE = mean_squared_error(test_pred_lst, labels_lst)
        test_Rp = pearsonr(test_pred_lst, labels_lst)
        return test_loss, test_MSE, test_Rp

