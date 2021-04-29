import torch
import torch.nn as nn
import numpy as np
import pdb

class PreferenceNet(nn.Module):
    def __init__(self, n_agents, n_items, hidden_dim):
        super(PreferenceNet, self).__init__()
        #size = n_agents * n_items
        size = n_agents * n_items + n_agents * n_items + n_agents
        
        self.MLP = nn.Sequential(nn.Linear(size, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), 
                                 nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1), nn.BatchNorm1d(1), nn.Sigmoid())

    def forward(self, bids, allocs, payments):
        bids = bids.view(bids.shape[0], -1)
        allocs = allocs.view(allocs.shape[0], -1)
        payments = payments.view(payments.shape[0], -1)
        
        #data = allocs
        data = torch.cat([bids, allocs, payments], dim=1)

        return self.MLP(data).squeeze(-1)