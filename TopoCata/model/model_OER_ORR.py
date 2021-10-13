# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
    
class TopoCata(torch.nn.Module):
    def __init__(self):
        super(TopoCata, self).__init__()
        self.edge_index = torch.tensor(
            [
                [3, 0, 3, 1, 3, 2, 3, 7, 7, 4, 7, 5, 7, 6, 0, 1, 1, 6, 6, 4, 4, 5, 5, 2, 2, 0], 
                [0, 3, 1, 3, 2, 3, 7, 3, 4, 7, 5, 7, 6, 7, 1, 0, 6, 1, 4, 6, 5, 4, 2, 5, 0, 2]
            ],
            dtype=torch.long
            )
        
        self.gcnc_0 = GCNConv(11, 64)
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
    
    
if __name__ == "__main__":
    inputs = torch.randn([1, 8, 11])
    net = TopoCata()
    outputs = net(inputs)
    print(outputs.size())
    