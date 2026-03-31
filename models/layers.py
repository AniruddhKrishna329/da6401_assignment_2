"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0<=p<=1
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p==0:
            return x
        mask=(torch.rand_like(x) > self.p).float()
        return mask*x/(1.0-self.p)
