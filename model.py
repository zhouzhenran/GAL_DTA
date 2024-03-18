from torch import nn
from layers import init_weight,Smiles1CNN

class DTA1_graph(nn.Module):
    def __init__(self, in_dims, emb_dim, hidden_dim, smi_hidden_dim):
        '''
        :param in_dims: 65
        :param emb_dim: 128
        :param smi_hidden_dims: 256
        '''
        super(DTA1_graph, self).__init__()
        self.hidden_dim = smi_hidden_dim
        self.smile_exa = Smiles1CNN(in_dim=in_dims,emb_dim=emb_dim,out_dim=smi_hidden_dim)
        self.hidden_layer = nn.Linear(smi_hidden_dim,smi_hidden_dim)
        self.outlayer = nn.Linear(smi_hidden_dim,1)
        self.relu = nn.ReLU()
        self.outlayer.apply(init_weight)

    def forward(self, data):
        smi_fea = self.smile_exa(data)
        smi_fea = self.relu(self.hidden_layer(smi_fea))
        out = self.outlayer(smi_fea)
        return out,smi_fea