from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import torch.optim as optim
from model.model import *

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
        testset = MyDataset(drug, target, label)
        test_loader = DataLoader(testset, batch_size=64, shuffle=False)

        return test_loader

    @torch.no_grad()
    def test(self,test_loader):
        dataset = pd.read_csv(self.DATAFILE)
        prelst = []
        explst = []
        for j in range(10):
        for j, path in enumerate(os.listdir('./data_process/cache')):
            model_path = os.path.join('./data_process/cache', path)
            #print(model_path)
            net.load_state_dict(torch.load(model_path))
            self.net.eval()
            for i, data in enumerate(test_loader):
                drug, target, labels = data[0], data[1], data[2]
                explst.append(labels.squeeze(1).view(-1, 1).numpy().squeeze())
                pred = self.net(drug, target)
                prelst.append(pred.squeeze(1).view(-1, 1).numpy().squeeze())

            prelst = np.array(prelst,dtype=object)
            prelst = np.concatenate(prelst,axis=0)
            col_name = f"pre{j+1}"

            dataset[col_name] = prelst
            explst = []
            prelst = []

        return dataset




if __name__ == "__main__":
    net = DTImodel()
    task = Task(net, "../dataset/4PDBbind_test.csv")
    test_loader = task.load_data()
    prelist = task.test(test_loader)
    prelist.to_csv("./4PDBbind_test.csv", index=False)
    
    task = Task(net, "../dataset/4PDBbind_train.csv")
    train_loader = task.load_data()
    prelist = task.test(train_loader)
    prelist.to_csv("./4PDBbind_train.csv", index=False)    
    
    print("test finishing")


