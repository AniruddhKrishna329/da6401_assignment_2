import torch
import torch.nn as nn
import gdown
import os
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, num_breeds:int=37, seg_classes:int=3, in_channels:int=3, classifier_path:str="checkpoints/classifier.pth", localizer_path:str="checkpoints/localizer.pth", unet_path:str="checkpoints/unet.pth"):
        super().__init__()
        os.makedirs("checkpoints",exist_ok=True)
        gdown.download(id="1Os0BF5mmsVvVvXKWDwk9EGcehD-NZb0z",output=classifier_path,quiet=False)
        gdown.download(id="1_zWV0ydt3ZMr-jc1HRwEVQxY3gisidWZ",output=localizer_path,quiet=False)
        gdown.download(id="1XMGSHNKW4qFK-glfSsW_TMsoOR_Wff3E",output=unet_path,quiet=False)

        classifier=VGG11Classifier(num_breeds,in_channels)
        classifier.load_state_dict(torch.load(classifier_path,map_location="cpu"))

        localizer=VGG11Localizer(in_channels)
        localizer.load_state_dict(torch.load(localizer_path,map_location="cpu"))

        unet=VGG11UNet(seg_classes,in_channels)
        unet.load_state_dict(torch.load(unet_path,map_location="cpu"))

        self.backbone=classifier.encoder
        self.cls_head=classifier.head
        self.loc_head=localizer.head
        self.seg_decoder=nn.ModuleDict({
            "up5":unet.up5,"dec5":unet.dec5,
            "up4":unet.up4,"dec4":unet.dec4,
            "up3":unet.up3,"dec3":unet.dec3,
            "up2":unet.up2,"dec2":unet.dec2,
            "up1":unet.up1,"dropout":unet.dropout,"final":unet.final})

    def forward(self, x:torch.Tensor):
        f5,feats=self.backbone(x,return_features=True)
        cls=self.cls_head(f5)
        loc=self.loc_head(f5)
        s=self.seg_decoder["up5"](f5)
        s=self.seg_decoder["dec5"](torch.cat([s,feats["f4"]],dim=1))
        s=self.seg_decoder["up4"](s)
        s=self.seg_decoder["dec4"](torch.cat([s,feats["f3"]],dim=1))
        s=self.seg_decoder["up3"](s)
        s=self.seg_decoder["dec3"](torch.cat([s,feats["f2"]],dim=1))
        s=self.seg_decoder["up2"](s)
        s=self.seg_decoder["dec2"](torch.cat([s,feats["f1"]],dim=1))
        s=self.seg_decoder["up1"](s)
        s=self.seg_decoder["dropout"](s)
        s=self.seg_decoder["final"](s)
        return {"classification":cls,"localization":loc,"segmentation":s}
