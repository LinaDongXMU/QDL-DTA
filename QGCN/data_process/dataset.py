import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from torch_geometric.data import Data, Dataset
from tqdm import tqdm


class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).sample(frac=1).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass
    
    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).sample(frac=1).reset_index()
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            graph = self._get_smiles2graph(mol['smiles'])
            label = self._get_labels(label=mol["label"])
            sequence = self._get_sequence(sequence=mol["sequence"])
            
        # Create data object
            data = Data(x = graph['node_feat'], 
                        edge_index = graph['edge_index'],
                        edge_attr = graph['edge_feat'],
                        num_node = graph['num_nodes'],
                        sequence = sequence,
                        y = label,
                        )
            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_test_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{index}.pt'))
    
    def _get_smiles2graph(self, smiles_string):
        mol = Chem.MolFromSmiles(smiles_string, sanitize=False)
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)

        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype = np.int64)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0: # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype = np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype = np.int64)

        else:   # mol has no bonds
            edge_index = np.empty((2, 0), dtype = np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

        graph = dict()
        graph['edge_index'] = torch.from_numpy(edge_index)
        graph['edge_feat'] = torch.from_numpy(edge_attr)
        graph['node_feat'] = torch.from_numpy(x)
        graph['num_nodes'] = len(x)

        return graph 

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.float32)
    
    def _get_sequence(self, sequence):
        VOCAB_PROTEIN = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                            "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                            "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                            "U": 19, "T": 20, "W": 21,
                            "V": 22, "Y": 23, "X": 24,
                            "Z": 25}

        targetint = [VOCAB_PROTEIN[s] for s in sequence]

        if len(targetint) < 1200:
            targetint = np.pad(targetint, (0, 1200 - len(targetint)))
        else:
            targetint = targetint[:1200]

        targetint = torch.tensor(targetint, dtype=torch.float32).unsqueeze(0)
        return targetint

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data
