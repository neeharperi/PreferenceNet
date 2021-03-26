import torch
import torch.nn as nn
import numpy as np
import pdb

class PreferenceNet(nn.Module):
    def __init__(self, n_agents, n_items, hidden_dim):
        super(PreferenceNet, self).__init__()
        size = n_agents * n_items
        self.MLP = nn.Sequential(nn.Linear(size, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, data):
        return self.MLP(data.view(data.shape[0], -1)).squeeze(-1)