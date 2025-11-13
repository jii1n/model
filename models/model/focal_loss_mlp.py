# focal_loss_mlp

from torch.nn import Sequential as Seq, Linear as Lin
from torch.nn import Module  
import torch.nn.functional as F
from torch.nn import BatchNorm1d, ReLU, Dropout
import torch


class FocalLoss(Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True, device='cpu'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma if gamma is not None else 2 
        self.device = device
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
   

class MLP(Seq):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, alpha, dropout, device='cpu'):
        print(f"[DEBUG] Received dropout in MLP init: {dropout}")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"[ERROR] Invalid dropout value in MLP __init__: {dropout}. Must be between 0 and 1.")
        self.dropout = dropout
        print(f"[DEBUG] Initializing MLP with dropout: {dropout}")
        
        layers = []
        layers.append(Lin(input_dim, hidden_dim))  


        for i in range(num_layers - 2):
            layers.append(BatchNorm1d(hidden_dim))
            layers.append(ReLU())
            if dropout > 0:
                layers.append(Dropout(dropout))
            if not (i == num_layers - 3):
                layers.append(Lin(hidden_dim, hidden_dim))


        layers.append(Lin(hidden_dim, output_dim))

        self.layers = layers
        super(MLP, self).__init__(*self.layers)


        self.focal_loss = FocalLoss(alpha,  device=device)


    def loss(self, pred, label):
        return self.focal_loss(pred, label)

    def weight_reset(self, m):
        if isinstance(m, Lin):
            m.reset_parameters()

    def forward(self, x):
        for module in self.layers:
            x = module(x)

        return x
