import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.layers import CustomDropout

class VGG11Localizer(nn.Module): #adds a bounding box around the pet's head. Outputs x_center, y_center, width, height.
    def __init__(self, in_channels:int=3, dropout_p:float=0.5):
        super().__init__()
        self.encoder=VGG11(in_channels)
        self.backbone_head=nn.Sequential(
            nn.AdaptiveAvgPool2d((7,7)),
            nn.Flatten(),
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p), #dropout layer with p=0.5
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p))
        self.bbox_head=nn.Linear(4096,4)

    def forward(self, x:torch.Tensor) -> torch.Tensor: #forward pass
        x=self.encoder(x)
        x=self.backbone_head(x)
        return torch.sigmoid(self.bbox_head(x))*224.0
