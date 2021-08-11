import torch
import torch.nn as nn
from torch import Tensor

class SubSpectralNorm(nn.Module):
    def __init__(self,
        num_features: int,
        num_subspecs: int = 2,
        eps: float = 1e-5,
        affine: bool = True
    ):
        super(SubSpectralNorm, self).__init__()
        self.eps = eps
        self.subpecs = num_subspecs
        self.gamma = nn.Parameter(torch.ones(num_features * num_subspecs),
            requires_grad=affine)
        self.beta = nn.Parameter(torch.zeros(num_features * num_subspecs),
            requires_grad=affine)
    
    def forward(self, x: Tensor):
        batch_size, num_channels, height, width = x.size()
        x = x.view(batch_size, num_channels*self.subpecs, height//self.subpecs,
            width)

        x_mean = x.mean([0, 2, 3]).view(1, num_channels * self.subpecs, 1, 1)
        x_var = x.var([0, 2, 3]).view(1, num_channels * self.subpecs, 1, 1)
        gamma = self.gamma.view(1, num_channels * self.subpecs, 1, 1)
        beta = self.beta.view(1, num_channels * self.subpecs, 1, 1)

        x = (x - x_mean) / (x_var + self.eps).sqrt() * gamma + beta

        return x.view(batch_size, num_channels, height, width)