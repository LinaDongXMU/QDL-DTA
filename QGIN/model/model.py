import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepquantum import Circuit
from deepquantum.gates.qcircuit import Circuit
from deepquantum.gates.qoperator import PauliX, PauliY, PauliZ
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import (BatchNorm, GINConv, MessagePassing,
                                global_mean_pool)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#——————————————Linear_layer—————————————————————————————————————————————————————————————————
class QuLinear(nn.Module):
    def __init__(self, input_dim, n_layers=5, measure=True):
        super().__init__()
        self.n_qubits = int(math.log(input_dim, 2))
        self.n_layers = n_layers
        self.weight = nn.Parameter(torch.empty((n_layers+1, self.n_qubits*3)))
        self.wires_lst = list(range(self.n_qubits))
        self.measure = measure
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.uniform_(self.weight, a=0.0, b=2 * math.pi)
        
    def amplitude_encoding(self, x):
        x = F.normalize(x, p=2, dim=-1) + 0j
        return x
    
    def add_variational_circuit(self):
        cir = Circuit(self.n_qubits)
        for i in range(self.n_layers):
            cir.XYZLayer(self.wires_lst, self.weight[i])
            cir.ring_of_cnot(self.wires_lst)
        cir.XYZLayer(self.wires_lst, self.weight[-1])
        return cir
    
    def measure_mps_batch(self, state, ith:list, pauli:str):
        assert state.ndim == self.n_qubits + 1, "ndim of input must be n_qubits+1"
        for i in range(1, self.n_qubits+1):
            assert state.shape[i] == 2, "ndim of component shape of input must be 2"
    
        measures = []
        for qubit in ith:
            mps = torch.clone(state)
            if (qubit >= self.n_qubits) or (qubit < 0):
                raise ValueError('qubit must less than n_qubits')
            if pauli.lower() == 'x':
                p_gate = PauliX(self.n_qubits, qubit)
            elif pauli.lower() == 'y':
                p_gate = PauliY(self.n_qubits, qubit)
            elif pauli.lower() == 'z':
                p_gate = PauliZ(self.n_qubits, qubit)
            mps = p_gate.TN_contract(mps, batch_mod=True)
            res = (state.reshape(state.shape[0], 1, -1).conj()) @ (mps.reshape(state.shape[0], -1, 1))
            measures.append(res.squeeze(-1).real)
        return torch.cat(measures, dim=-1)

    def forward(self, x):
        U_encoding = self.amplitude_encoding(x)
        cir = self.add_variational_circuit()
        MPS = U_encoding.view([x.shape[0]] + [2] * self.n_qubits)
        final_state = cir.TN_contract_evolution(MPS, batch_mod=True)
        if self.measure:
            exp_x = self.measure_mps_batch(final_state, list(range(self.n_qubits)), 'x')
            exp_y = self.measure_mps_batch(final_state, list(range(self.n_qubits)), 'y')
            exp_z = self.measure_mps_batch(final_state, list(range(self.n_qubits)), 'z')
            return torch.cat([exp_x, exp_y, exp_z], dim=-1)
        final_state = final_state.reshape(x.shape[0], 1, -1)
        return final_state.squeeze(1).real


class Regression_Layer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.qulinear_out = 3*int(math.log(hidden_dim, 2))
        self.dense = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            QuLinear(hidden_dim, measure=True),
            nn.Linear(self.qulinear_out,1),
        )
    def forward(self,x):
        out = self.dense(x)
        return out
#——————————————end——————————————————————————————————————————————————————————————————————————

#——————————————drug_net—————————————————————————————————————————————————————————————————————
class GINConv(MessagePassing):
    def __init__(self, emb_dim, mlp):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(GINConv, self).__init__(aggr = "add")

        self.mlp = mlp
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr) # First, convert the category edge attribute to edge representation
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        #`message() ` X in function_ j + edge_ The attr ` operation performs the fusion of node information and edge information
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out):
        return aggr_out
    
    
class DrugNet(torch.nn.Module):
    def __init__(self, in_dim, num_layers, drop_ratio, JK='sum'):
        super(DrugNet, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.node_emb = AtomEncoder(in_dim)
        self.mlp = nn.Sequential(QuLinear(in_dim, measure=False), nn.BatchNorm1d(in_dim), nn.ReLU(), QuLinear(in_dim, measure=False))
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            conv = GINConv(in_dim, self.mlp)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(in_dim))
        
    def forward(self, x, edge_index, edge_attr, batch):
        h_lst = [self.node_emb(x)]
        for layer in range(self.num_layers):
            h = self.convs[layer](x=h_lst[layer], edge_index=edge_index, edge_attr=edge_attr)
            h = self.batch_norms[layer](h)
        
            if layer == self.num_layers-1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)       
            h_lst.append(h)
              
        if self.JK == "last":
            node_representation = h_lst[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers + 1):
                node_representation += h_lst[layer]

        output = global_mean_pool(node_representation, batch)
        return output
#——————————————end——————————————————————————————————————————————————————————————————————————

#——————————————target_net———————————————————————————————————————————————————————————————————
class SequenceNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, filter_num, kernel_lst):
        super(SequenceNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.conv1 = nn.Conv1d(emb_dim, filter_num, kernel_lst[0], stride=1)
        self.conv2 = nn.Conv1d(filter_num, 2*filter_num, kernel_lst[1], stride=1)
        self.conv3 = nn.Conv1d(2*filter_num, 3*filter_num, kernel_lst[2], stride=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x):
        x = x.to(torch.int32)
        x = self.embedding(x).permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        return x
#——————————————end——————————————————————————————————————————————————————————————————————————

#——————————————DTI_structure————————————————————————————————————————————————————————————————
class DTINet(nn.Module):
    def __init__(self):
        super().__init__()
        self.drugquanv = DrugNet(in_dim=64, num_layers=1, drop_ratio=0.3, JK='sum')
        self.targetquanv = SequenceNet(vocab_size=25, emb_dim=128, filter_num=32, kernel_lst=[4,6,8])
        self.linearlayer = Regression_Layer(160, 1024)
    
    def forward(self, node_feature, edge_index, edge_attr, target_input, batch):
        drug_output = self.drugquanv(node_feature, edge_index, edge_attr, batch)     # 64 16
        target_output = self.targetquanv(target_input)   #64 16 16
        linear_input = torch.cat([drug_output, target_output],dim=-1)
        linear_output = self.linearlayer(linear_input)
        linear_output = linear_output.squeeze(-1)
        return linear_output
#——————————————end——————————————————————————————————————————————————————————————————————————
