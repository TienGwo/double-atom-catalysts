# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class AtomsFeat(torch.nn.Module):
    def __init__(self, feature_num):
        super(AtomsFeat, self).__init__()
        self.regression = nn.Sequential(
            nn.Linear(feature_num, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
            )
    
    def forward(self, x):
        x = self.regression(x)
        return x
    
    
class SubstrateFeat(torch.nn.Module):
    def __init__(self, feature_num):
        super(SubstrateFeat, self).__init__()
        self.regression = nn.Sequential(
            nn.Conv1d(feature_num, 64, 3),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3),
            nn.ReLU(),
            nn.Conv1d(128, 1, 2)
            )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.regression(x)
        return x.squeeze(2)
    
    
class GraphFeat(torch.nn.Module):
    def __init__(self, feature_num, map_location):
        super(GraphFeat, self).__init__()
        self.edge_index = torch.tensor(
            [
                [3, 0, 3, 1, 3, 2, 3, 7, 7, 4, 7, 5, 7, 6, 0, 1, 1, 6, 6, 4, 4, 5, 5, 2, 2, 0], 
                [0, 3, 1, 3, 2, 3, 7, 3, 4, 7, 5, 7, 6, 7, 1, 0, 6, 1, 4, 6, 5, 4, 2, 5, 0, 2]
            ],
            dtype=torch.long
            )
        
        if map_location == "gpu":
            self.edge_index = self.edge_index.cuda()
        
        self.gcnc_0 = GCNConv(feature_num, 64)
        self.gcnc_1 = GCNConv(64, 96)
        self.gcnc_2 = GCNConv(96, 128)
        
        self.regression = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1),
            nn.AdaptiveMaxPool1d(1)
            )
        
    def forward(self, x):
        x = F.relu(self.gcnc_0(x, self.edge_index))
        x = F.relu(self.gcnc_1(x, self.edge_index))
        x = F.relu(self.gcnc_2(x, self.edge_index))
        x = x.transpose(1, 2)
        x = self.regression(x)
        x = x.squeeze(2)
        return x
    
    
class TopoCata(torch.nn.Module):
    def __init__(self, feature_num, map_location="gpu"):
        super(TopoCata, self).__init__()
        self.atoms_feature = AtomsFeat(feature_num)
        self.graph_feature = GraphFeat(feature_num, map_location)
        self.substrate_feature = SubstrateFeat(feature_num)
        self.global_feature = nn.Sequential(
            nn.Linear(3, 64, 1),
            nn.ReLU(),
            nn.Linear(64, 128, 1),
            nn.ReLU(),
            nn.Linear(128, 1, 1)
            )
        
    def forward(self, x):
        substrate_feature = self.substrate_feature(x[:, [0, 1, 2, 4, 5, 6], :])
        atoms_feature = self.atoms_feature(x[:, 3, :] + x[:, 7, :])
        graph_feature = self.graph_feature(x)
        feature = torch.cat([atoms_feature, graph_feature, substrate_feature], dim=1)
        feature = self.global_feature(feature)
        return feature
    
    
if __name__ == "__main__":
    inputs = torch.randn([16, 8, 11]).cuda().float()
    net = TopoCata(feature_num=inputs.shape[2]).cuda()
    outputs = net(inputs)
    print(outputs.size())
    