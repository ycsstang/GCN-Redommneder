#codebase
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data,HeteroData
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv,HeteroConv
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split


from utlis import create_graph, prepare_loaders
from model import GNNEncoder, EdgeDecoder,Model

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
device = 'cuda' if torch.cuda.is_available() else 'cpu'




# 划分训练集和测试集
train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)
train_graph = create_graph(train_ratings).to(device)
test_graph = create_graph(test_ratings).to(device)
print("用户节点数:", train_graph['user'].x.shape[0])
print("电影节点数:", train_graph['movie'].x.shape[0])
print("边索引最大值 - user:", train_graph['user', 'rates', 'movie'].edge_index[0].max().item())
print("边索引最大值 - movie:", train_graph['user', 'rates', 'movie'].edge_index[1].max().item())
user_feat_dim = train_graph['user'].x.size(1)
movie_feat_dim = train_graph['movie'].x.size(1)

train_loader, test_loader = prepare_loaders(train_graph, test_graph, batch_size=4096)

model = Model(hidden_channels=64, metadata=train_graph.metadata()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
criterion = torch.nn.MSELoss()



def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred = model(batch.x_dict, batch.edge_index_dict, 
                    batch[('user', 'rates', 'movie')].edge_label_index)
        
        # 使用均方误差损失
        loss = F.mse_loss(pred, batch[('user', 'rates', 'movie')].edge_label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.size(0)
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    total_mse = 0
    total_mae = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x_dict, batch.edge_index_dict,
                        batch[('user', 'rates', 'movie')].edge_label_index)
            
            mse = F.mse_loss(pred, batch[('user', 'rates', 'movie')].edge_label)
            mae = F.l1_loss(pred, batch[('user', 'rates', 'movie')].edge_label)
            
            total_mse += float(mse) * pred.size(0)
            total_mae += float(mae) * pred.size(0)
    
    return {
        'mse': total_mse / len(loader.dataset),
        'mae': total_mae / len(loader.dataset),
        'rmse': np.sqrt(total_mse / len(loader.dataset))
    }


def generate_recommendations(model, data, user_id, top_k=10, device='cpu'):
    """
    为指定用户生成电影推荐
    
    参数:
        model: 训练好的推荐模型
        data: 包含图数据的HeteroData对象
        user_id: 要推荐的用户ID(原始ID或编码后ID)
        top_k: 返回的推荐数量
        device: 计算设备('cpu'或'cuda')
    
    返回:
        tuple: (推荐电影ID列表, 预测评分列表)
    """
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        # 获取所有节点表示
        z_dict = model.encoder(data.x_dict, data.edge_index_dict)
        
        # 确保user_id是tensor格式
        if isinstance(user_id, int):
            user_id = torch.tensor([user_id], device=device)
        
        # 获取目标用户表示
        user_emb = z_dict['user'][user_id]  # shape: [1, emb_dim]
        movie_emb = z_dict['movie']         # shape: [num_movies, emb_dim]
        
        # 计算用户对所有电影的预测分数 (使用解码器)
        edge_label_index = torch.stack([
            user_id.repeat(movie_emb.size(0)),
            torch.arange(movie_emb.size(0), device=device)
        ])
        scores = model.decoder(z_dict, edge_label_index)  # shape: [num_movies]
        
        # 排除已评分的电影
        rated_mask = torch.zeros(movie_emb.size(0), dtype=torch.bool, device=device)
        user_edges = (data['user', 'rates', 'movie'].edge_index[0] == user_id)
        rated_movies = data['user', 'rates', 'movie'].edge_index[1][user_edges]
        rated_mask[rated_movies] = True
        scores[rated_mask] = -float('inf')
        
        # 获取top-k推荐及其分数
        top_scores, top_movies = torch.topk(scores, k=min(top_k, len(scores)))
        
        return top_movies.cpu().numpy(), top_scores.cpu().numpy()
    




# 训练循环
print("开始训练...")
# 训练循环
for epoch in range(1, 6):
    train_loss = train(model, train_loader, optimizer, device)
    test_metrics = evaluate(model, test_loader, device)
    
    scheduler.step(epoch)
    
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
          f'Test RMSE: {test_metrics["rmse"]:.4f}, MAE: {test_metrics["mae"]:.4f}')
    
# 保存模型
torch.save(model.state_dict(), 'movie_recommender.pth')


# 示例推荐
user_id = 0  # 选择第一个用户
id_list,score_list = generate_recommendations(model, train_graph, 50,device=device)

movie_titles = movies['title'].to_dict()

print(f"\n为用户 {50} 推荐的电影:")
for i in range(10):
    original_id = id_list[i]
    print(f"{i+1}. {movie_titles.get(original_id, '未知电影')}  score:{score_list[i]:.5f}")