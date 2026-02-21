import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x): return self.net(x)

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = float(dropout)
    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = float(dropout)
    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)
