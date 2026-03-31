"""Classification components
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.layers import CustomDropout
class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder=VGG11(in_channels)
        self.head=nn.Sequential(
            nn.AdaptiveAvgPool2d((7,7)),
            nn.Flatten(),
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096,num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       x=self.encoder(x)
       return self.head(x)
