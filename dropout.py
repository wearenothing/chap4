import torch
from d2l import torch as d2l
from torch import nn

def dropout_layer(X, dropout):
    assert 0<=dropout<=1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    # mask = (torch.rand(X.shape)>dropout).float()
    mask = (torch.rand(X.shape)>dropout).float()
    return mask*X / (1-dropout)

X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))