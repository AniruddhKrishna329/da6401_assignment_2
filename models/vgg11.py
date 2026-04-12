import torch
import torch.nn as nn
from typing import Dict,Tuple,Union

class VGG11(nn.Module): #backbone feature extractor for all three models.
    def __init__(self, in_channels:int=3, use_bn:bool=True):
        super().__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(in_channels,64,3,padding=1),
            nn.BatchNorm2d(64) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2))
        self.block2=nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2))
        self.block3=nn.Sequential(
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2))
        self.block4=nn.Sequential(
            nn.Conv2d(256,512,3,padding=1),
            nn.BatchNorm2d(512) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2))
        self.block5=nn.Sequential(
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2))
    def forward(self, x:torch.Tensor, return_features:bool=False) -> Union[torch.Tensor,Tuple[torch.Tensor,Dict[str,torch.Tensor]]]:
        f1=self.block1(x)
        f2=self.block2(f1)
        f3=self.block3(f2)
        f4=self.block4(f3)
        f5=self.block5(f4)
        if return_features:
            return f5,{"f1":f1,"f2":f2,"f3":f3,"f4":f4,"f5":f5}
        return f5

VGG11Encoder=VGG11
