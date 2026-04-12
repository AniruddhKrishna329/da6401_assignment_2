import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.layers import CustomDropout

class VGG11Localizer(nn.Module):
    def __init__(self, in_channels:int=3, dropout_p:float=0.5):
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
            nn.Linear(4096,4))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x=self.encoder(x)
        out=self.head(x)
        # scale to image space: cx,cy in [0,224], w,h in [0,224]
        out=torch.sigmoid(out)*224
        return out
