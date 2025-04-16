#codebase
import pandas as pd
import torch
from torch_geometric.data import Data,HeteroData
from torch_geometric.loader import LinkNeighborLoader



movies = pd.read_csv('movies.csv')
def create_graph(df):
    # 对 userId 和 movieId 重新编码为从 0 开始的连续整数
    df = df.copy()
    df['userId'] = df['userId'].astype('category').cat.codes
    df['movieId'] = df['movieId'].astype('category').cat.codes

    num_users = df['userId'].nunique()
    num_movies = df['movieId'].nunique()
    graph = HeteroData()
    
    # 确保 user.x 和 movie.x 的维度足够大
    graph['user'].x = torch.arange(num_users, dtype=torch.float).view(-1, 1)

    merged_df = pd.merge(df,movies,how='left',on='movieId')
    merged_df = merged_df[['movieId','genres']].drop_duplicates()
    genres = merged_df['genres'].str.get_dummies(sep='|')
    item_features = torch.tensor(genres.values, dtype=torch.float)
    graph['movie'].x = item_features
    
    # 检查 edge_index 的索引范围
    edge_index = torch.tensor([df['userId'].values, df['movieId'].values], dtype=torch.long)
    
    assert edge_index[0].max() < num_users, "User ID 越界"
    assert edge_index[1].max() < num_movies, "Movie ID 越界"
    
    graph['user', 'rates', 'movie'].edge_index = edge_index
    graph['user', 'rates', 'movie'].edge_attr = torch.tensor(df['rating'].values, dtype=torch.float) 
    graph['movie', 'rated_by', 'user'].edge_index = edge_index[[1, 0]]  # 更安全的翻转方式
    
    return graph



def prepare_loaders(train_graph, test_graph, batch_size=512):
    # 训练加载器 - 包含评分作为边属性
    train_loader = LinkNeighborLoader(
        data=train_graph,
        num_neighbors=[20, 10],  # 采样邻居数
        edge_label_index=(('user', 'rates', 'movie'), train_graph[('user', 'rates', 'movie')].edge_index),
        edge_label=train_graph[('user', 'rates', 'movie')].edge_attr,  # 使用实际评分
        batch_size=batch_size,
        shuffle=True,
    )
    
    # 测试加载器
    test_loader = LinkNeighborLoader(
        data=test_graph,
        num_neighbors=[20, 10],
        edge_label_index=(('user', 'rates', 'movie'), test_graph[('user', 'rates', 'movie')].edge_index),
        edge_label=test_graph[('user', 'rates', 'movie')].edge_attr,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader