import torch
import torch.nn as nn
from models.vgg11 import VGG11
from models.layers import CustomDropout

class VGG11UNet(nn.Module):
    def __init__(self, num_classes:int=3, in_channels:int=3, dropout_p:float=0.5):
        super().__init__()
        self.encoder=VGG11(in_channels)

        self.up5=nn.ConvTranspose2d(512,512,2,stride=2)
        self.dec5=nn.Sequential(
            nn.Conv2d(1024,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.up4=nn.ConvTranspose2d(512,256,2,stride=2)
        self.dec4=nn.Sequential(
            nn.Conv2d(512,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.up3=nn.ConvTranspose2d(256,128,2,stride=2)
        self.dec3=nn.Sequential(
            nn.Conv2d(256,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.up2=nn.ConvTranspose2d(128,64,2,stride=2)
        self.dec2=nn.Sequential(
            nn.Conv2d(128,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.up1=nn.ConvTranspose2d(64,32,2,stride=2)
        self.dropout=CustomDropout(dropout_p)
        self.final=nn.Conv2d(32,num_classes,1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        f5,feats=self.encoder(x,return_features=True)

        x=self.up5(f5)
        x=self.dec5(torch.cat([x,feats["f4"]],dim=1))

        x=self.up4(x)
        x=self.dec4(torch.cat([x,feats["f3"]],dim=1))

        x=self.up3(x)
        x=self.dec3(torch.cat([x,feats["f2"]],dim=1))

        x=self.up2(x)
        x=self.dec2(torch.cat([x,feats["f1"]],dim=1))

        x=self.up1(x)
        x=self.dropout(x)
        return self.final(x)