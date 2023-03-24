import torch
import torch.nn as nn
import torch.nn.functional as F
    
    
class MLP(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 dropout=0,
                 batch_norm=False,
                 num_layers=2,
                 activation=F.relu):
        super().__init__()
        assert num_layers >= 2
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_dim, hid_dim))
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bns = nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(hid_dim))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hid_dim, hid_dim))
            if self.batch_norm:
                self.bns.append(nn.BatchNorm1d(hid_dim))
        self.lins.append(nn.Linear(hid_dim, out_dim))

        self.dropout = nn.Dropout(dropout)
        self.act_fn = activation
        
    def forward(self, x):    
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.batch_norm:
                x = self.bns[i](x)
            x = self.act_fn(x)
            x = self.dropout(x)
        x = self.lins[-1](x)
        return x


class MLPLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPLinear, self).__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x):
        return self.lin(x)