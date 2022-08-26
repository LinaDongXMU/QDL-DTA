import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
from deepquantum import Circuit



class SeqRepresentation(nn.Module):
    def __init__(self, vocab_size, embedding_num, kernel_sizes, filter_num):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.conv1 = nn.Conv1d(embedding_num, filter_num, kernel_size=kernel_sizes[0], stride=1)
        self.conv2 = nn.Conv1d(filter_num, filter_num * 2, kernel_size=kernel_sizes[1], stride=1)
        self.conv3 = nn.Conv1d(filter_num * 2, filter_num * 3, kernel_size=kernel_sizes[2], stride=1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)

        return x


class QuLinear(nn.Module):
    """
    quantum linear layer
    """

    def __init__(self, input_dim, n_layers, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super().__init__()

        he_std = gain * 5 ** (-0.5)
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.input_dim = input_dim
        self.n_qubits = int(math.log(input_dim, 2))
        self.N3 = 3 * self.n_qubits
        self.dim = (1 << self.n_qubits)  # 2**n_qubits

        self.n_layers = n_layers

        self.n_param = self.N3 * (self.n_layers + 1)
        self.weight = nn.Parameter(
            nn.init.uniform_(torch.empty(self.n_param), a=0.0, b=2 * np.pi) * init_std * self.w_mul)

        self.cir = Circuit(self.n_qubits)
        self.wires_lst = list(range(self.n_qubits))
        self.is_batch = False

    def encoding_layer(self, x):
        out = F.normalize(x, dim=-1) + 0j
        return out

    def forward(self, x):
        x = x.unsqueeze(1)
        self.cir.clear()

        if x.ndim == 3:
            assert (x.shape[1] == 1) and (x.shape[2] <= self.dim), "批处理情况时，输入数据的1轴数据长度为1、2轴数据长度不超过2的比特数次幂"
            if x.shape[2] < self.dim:
                pad = nn.ZeroPad2d(padding=(0, self.dim - x.shape[2], 0, 0))
                x = pad(x)
            self.is_batch = True
            x = self.encoding_layer(x)
        else:
            raise ValueError("输入数据的维度大小限定为3(批处理)")

        for i in range(self.n_layers):
            index = i * self.N3
            self.cir.XYZLayer(self.wires_lst, self.weight[index: index + self.N3])
            self.cir.ring_of_cnot(self.wires_lst)
        index += self.N3
        self.cir.YZYLayer(self.wires_lst, self.weight[index:])

        if self.is_batch:
            x = x.view([x.shape[0]] + [2] * self.n_qubits)
            res = self.cir.TN_contract_evolution(x, batch_mod=True)
            res = res.reshape(res.shape[0], 1, -1)
            assert res.shape[2] == self.dim, "线性层MPS演化结果数据2轴数据大小要求为2的比特数次幂"
        else:
            # x = nn.functional.normalize(x, dim=1)
            x = self.cir.state_init()
            x = x.view([2] * self.n_qubits)
            res = self.cir.TN_contract_evolution(x, batch_mod=False)
            res = res.reshape(1, -1)
            assert res.shape[1] == self.dim, "线性层MPS演化结果数据1轴数据大小要求为2的比特数次幂"

        # return res.squeeze(1).real

        return res.real
#

class DTImodel(nn.Module):
    def __init__(self, vocab_ligand_size=64, vocab_protein_size=25, embedding_size=128, filter_num=32):
        super().__init__()
        self.protein_encoder = SeqRepresentation(vocab_protein_size, embedding_size, kernel_sizes=[4, 8, 12],
                                                 filter_num=filter_num)
        self.ligand_encoder = SeqRepresentation(vocab_ligand_size, embedding_size, kernel_sizes=[4, 6, 8],
                                                filter_num=filter_num)
        self.qlinear = QuLinear(256, 4)
        # self.linear1 = nn.Linear(filter_num * 3 * 2, 1024)
        # self.drop1 = nn.Dropout(0.1)
        # self.linear2 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(256, 1024)
        self.drop2 = nn.Dropout(0.1)
        self.linear3 = nn.Linear(1024, 512)
        self.drop3 = nn.Dropout(0.1)
        self.out_layer = nn.Linear(512, 1)

    def forward(self, ligand_x, protein_x):
        protein_x = self.protein_encoder(protein_x)
        ligand_x = self.ligand_encoder(ligand_x)
        x = torch.cat([protein_x, ligand_x], dim=-1)
        x = F.normalize(x, dim=-1) + 0j
        x =self.qlinear(x)
        # x = F.relu(self.linear1(x))
        # x = self.drop1(x)
        x = F.relu(self.linear2(x))
        x = self.drop2(x)
        x = F.relu(self.linear3(x))
        x = self.drop3(x)
        x = self.out_layer(x)

        return x







