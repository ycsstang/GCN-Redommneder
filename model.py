import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv,HeteroConv
from torch_geometric.nn import global_mean_pool
import torch
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict({
            'user': torch.nn.LazyLinear(hidden_channels),
            'movie': torch.nn.LazyLinear(hidden_channels)
        })
        
        self.norm_dict = torch.nn.ModuleDict({
            'user': torch.nn.BatchNorm1d(hidden_channels),
            'movie': torch.nn.BatchNorm1d(hidden_channels)
        })
        
        self.dropout = torch.nn.Dropout(0.2)
        
        self.conv1 = HeteroConv({
            ('user', 'rates', 'movie'): SAGEConv((-1, -1), hidden_channels),
            ('movie', 'rated_by', 'user'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('user', 'rates', 'movie'): SAGEConv((-1, -1), out_channels),
            ('movie', 'rated_by', 'user'): SAGEConv((-1, -1), out_channels),
        }, aggr='mean')
    
    def forward(self, x_dict, edge_index_dict):
        # 特征变换
        x_dict = {key: self.lin_dict[key](x) for key, x in x_dict.items()}
        x_dict = {key: self.norm_dict[key](x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(F.relu(x)) for key, x in x_dict.items()}
        
        # 图卷积
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: self.dropout(F.relu(x)) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        
        return x_dict

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, 4 * hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(4 * hidden_channels, 2 * hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_channels, 1)
        )
    
    def forward(self, z_dict, edge_label_index):
        user_z = z_dict['user'][edge_label_index[0]]
        movie_z = z_dict['movie'][edge_label_index[1]]
        features = torch.cat([user_z, movie_z], dim=-1)
        return self.mlp(features).squeeze(-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.decoder = EdgeDecoder(hidden_channels)
    
    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)