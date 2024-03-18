import torch
from torch import nn
from torch.nn import Parameter
from typing import *
from torch_geometric.nn import MessagePassing,GATConv,GINConv,GATv2Conv,GraphConv,GCNConv
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool as gap, global_mean_pool as gmp, global_max_pool, TopKPooling,SAGPooling
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch_geometric.nn.inits import glorot, zeros
from enum import IntEnum
class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor):

        return input.squeeze(self.dim)


class Smiles1CNN(nn.Module):
    def __init__(self, in_dim=65, emb_dim=128, out_dim=128):
        '''

        :param in_dim: 65
        :param emb_dim:
        :param out_dim:
        '''
        super(Smiles1CNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=in_dim, embedding_dim=emb_dim)
        hidden_size = out_dim//2
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size,batch_first=True,bidirectional=True,num_layers=1)
        self.gat = GAT1Block(out_dim=out_dim)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)
        self.relu = nn.LeakyReLU(0.1)

        self.fusion = FusionBlock(in_dim=out_dim,r=4,num_head=4)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)


    def forward(self,data):
        x = data.smi_emb
        # x [n,l]
        x_emb = self.embedding(x)  # [n,l,emb_dim]
        out1, _ = self.lstm(x_emb)  # [n,l,o_d)
        out1 = torch.transpose(out1, 1, 2)  # [n,o_d,l)
        out1 = self.pool(out1)  # [n,o_d,1]
        out1 = self.squeeze(out1)  # [n,o_d]

        out2 = self.gat(data)

        out = self.fusion(out1,out2)

        return out


class GAT1Block(nn.Module):
    def __init__(self,num_dim=78,out_dim=128,heads=5,dropout=0.2):
        super(GAT1Block, self).__init__()

        self.gat1 = GATv2Conv(num_dim,num_dim,heads=heads,dropout=0.2)
        self.gat2 = GATv2Conv(num_dim*heads,out_dim,dropout=0.2)

        self.fc = nn.Linear(out_dim,out_dim)

        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)
        self.fc.apply(init_weight)


    def forward(self,data):
        x,edge_index,batch = data.x, data.edge_index,data.batch
        x = x.float()

        x_1 = self.gat1(x,edge_index)
        x_1 = self.relu(x_1)
        x_2 = self.gat2(x_1,edge_index)
        x = self.relu(x_2)

        x = global_max_pool(x,batch)
        # x = gap(x,batch)
        x = self.fc(x)
        x = self.relu(x)

        return x


def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight)

class FusionBlock(torch.nn.Module):
    def __init__(self, in_dim, r=2, num_head=4):
        super(FusionBlock, self).__init__()
        self.data_dim = in_dim
        self.hidden_dim = int(self.data_dim//r)
        # 局部特征融合
        self.local_att = nn.Sequential(
            nn.Linear(self.data_dim,self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.data_dim),
            nn.BatchNorm1d(self.data_dim),
            nn.LeakyReLU(0.1)
        )
        self.local_att.apply(init_weight)
        # 全局特征融合
        self.global_att = nn.MultiheadAttention(embed_dim=self.data_dim,num_heads=num_head,dropout=0.1)
        nn.init.xavier_uniform_(self.global_att.in_proj_weight)

        self.sigmoid = nn.Sigmoid()

        # 第二次局部特征融合
        self.local_att2 = nn.Sequential(
            nn.Linear(self.data_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.data_dim),
            nn.BatchNorm1d(self.data_dim),
            nn.LeakyReLU(0.1)
        )
        self.local_att2.apply(init_weight)
        # 第二层全局特征融合
        self.global_att2 = nn.MultiheadAttention(embed_dim=self.data_dim,num_heads=num_head,dropout=0.1)
        nn.init.xavier_uniform_(self.global_att2.in_proj_weight)

    def forward(self, x, y):
        f1 = x + y
        hl1 = self.local_att(f1)
        f2 = f1.unsqueeze(0) #[N,dim]-[1,N,dim]
        hg1,_ = self.global_att(f2,f2,f2)
        hg1 = hg1.squeeze(0)#[1,N,dim]-[N,dim]

        h1 = hl1 + hg1
        weights1 = self.sigmoid(h1)
        x1 = x * weights1 + y * (1-weights1)

        hl2 = self.local_att2(x1)
        f3 = x1.unsqueeze(0)
        hg2,_ = self.global_att2(f3,f3,f3)
        hg2 = hg2.squeeze(0)

        h2 = hl2 + hg2
        weights2 = self.sigmoid(h2)
        x2 = x * weights2 + y * (1-weights2)

        return x2
