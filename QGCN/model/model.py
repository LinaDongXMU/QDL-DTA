import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import BatchNorm, MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
import math
from deepquantum import Circuit
import numpy as np


def data_processing(x):
    data_list = []
    for i in x:
        i = np.pad(i, (0, 16 - len(i)))
        # i = i[0:1024]
        data_list.append(i.tolist())

    return torch.tensor(data_list)


# ——————————————Linear_layer———————————————————————————————————————————————————————————————————
class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_net = nn.Sequential(
            nn.Linear(112, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        out = self.fc_net(x)
        return out


# ——————————————end————————————————————————————————————————————————————————————————————————————

# ——————————————drug_net(GAT)——————————————————————————————————————————————————————————————————
class Q_Liner(nn.Module):

    def __init__(self, n_qubits, n_layers=5, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()
        he_std = gain * 5 ** (-0.5)
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        # self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.N3 = 3 * self.n_qubits
        self.dim = (1 << self.n_qubits)  # 2**n_qubits
        self.n_layers = n_layers
        self.n_param = self.N3 * (self.n_layers + 1)
        self.weight = nn.Parameter(
            nn.init.uniform_(torch.empty(self.n_param), a=0.0, b=2 * torch.pi) * init_std * self.w_mul)
        self.cir = Circuit(self.n_qubits)
        self.wires_lst = list(range(self.n_qubits))
        self.is_batch = False

    def amplitude_encoding(self, x):
        x = x.clone().detach()
        norm = torch.sqrt(torch.sum(x * torch.conj(x), dim=-1, keepdim=True))

        out = x / torch.sqrt(norm + 1e-12) + 0j
        out = out.unsqueeze(1)
        return out

    def forward(self, x):
        x = self.amplitude_encoding(x)
        self.cir.clear()
        self.is_batch = True
        for i in range(self.n_layers):
            index = i * self.N3
            self.cir.XYZLayer(self.wires_lst, self.weight[index: index + self.N3])
            self.cir.ring_of_cnot(self.wires_lst)
        index += self.N3
        self.cir.YZYLayer(self.wires_lst, self.weight[index:])
        x = x.reshape([x.shape[0]] + [2] * self.n_qubits)
        res = self.cir.TN_contract_evolution(x, batch_mod=True)
        res = res.reshape(res.shape[0], 1, -1).squeeze(1)
        assert res.shape[1] == self.dim, "线性层MPS演化结果数据2轴数据大小要求为2的比特数次幂"

        return res.real


class GCNConv(MessagePassing):
    def __init__(self, n_qubits):
        super(GCNConv, self).__init__(aggr='add')
        self.fc = Q_Liner(n_qubits)

    def forward(self, x, edge_index):
        
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.fc(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)


class DrugNet(torch.nn.Module):
    def __init__(self):
        super(DrugNet, self).__init__()

        self.conv1 = GCNConv(4)
        self.conv2 = GCNConv(4)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = F.log_softmax(x, dim=1)
        x = global_mean_pool(x, batch)

        return x


# ——————————————end——————————————————————————————————————————————————————————————————————————

# ——————————————target_net———————————————————————————————————————————————————————————————————
class SequenceNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, filter_num, kernel_lst):
        super(SequenceNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.conv1 = nn.Conv1d(emb_dim, filter_num, kernel_lst[0], stride=1)
        self.conv2 = nn.Conv1d(filter_num, 2 * filter_num, kernel_lst[1], stride=1)
        self.conv3 = nn.Conv1d(2 * filter_num, 3 * filter_num, kernel_lst[2], stride=1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = x.to(torch.int32)
        x = self.embedding(x).permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        return x


class PocketNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, filter_num, kernel_lst):
        super(PocketNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.conv1 = nn.Conv1d(emb_dim, filter_num, kernel_lst[0], stride=1)
        self.conv2 = nn.Conv1d(filter_num, 2 * filter_num, kernel_lst[1], stride=1)
        self.conv3 = nn.Conv1d(2 * filter_num, 3 * filter_num, kernel_lst[2], stride=1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = x.to(torch.int32)
        x = self.embedding(x).permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        return x


# ——————————————end——————————————————————————————————————————————————————————————————————————

# ——————————————DTI_structure————————————————————————————————————————————————————————————————
class DTINet(nn.Module):  
    def __init__(self):
        super().__init__()
        self.drugquanv = DrugNet()
        self.sequencequanv = SequenceNet(vocab_size=25, emb_dim=128, filter_num=32, kernel_lst=[4, 6, 8])
        # self.pocketquanv = PocketNet(vocab_size=25, emb_dim=128, filter_num=32, kernel_lst=[4,6,8])
        self.linearlayer = Linear()

    def forward(self, node_feature, edge_index, sequence_input, batch):
        node_feature = data_processing(node_feature)
        drug_output = self.drugquanv(node_feature, edge_index, batch)  # 64 16
        sequence_output = self.sequencequanv(sequence_input)  # 64 16 16
        # pocket_output = self.pocketquanv(pocket_input)
        linear_input = torch.cat([drug_output, sequence_output], dim=-1)
        linear_output = self.linearlayer(linear_input)
        linear_output = linear_output.squeeze(-1)
        return linear_output
# ——————————————end——————————————————————————————————————————————————————————————————————————
