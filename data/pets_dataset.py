import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class OxfordIIITPetDataset(Dataset):
    def __init__(self, root:str="data", split:str="trainval", transform=None):
        self.root=root
        self.transform=transform
        self.img_dir=os.path.join(root,"images")
        self.ann_dir=os.path.join(root,"annotations")
        self.xml_dir=os.path.join(self.ann_dir,"xmls")
        self.mask_dir=os.path.join(self.ann_dir,"trimaps")

        split_file=os.path.join(self.ann_dir,f"{split}.txt")
        self.samples=[]
        with open(split_file) as f:
            for line in f:
                parts=line.strip().split()
                if len(parts)<2:
                    continue
                name=parts[0]
                class_id=int(parts[1])-1
                self.samples.append((name,class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name,class_id=self.samples[idx]
        img=Image.open(os.path.join(self.img_dir,f"{name}.jpg")).convert("RGB")
        mask=Image.open(os.path.join(self.mask_dir,f"{name}.png")).resize((224,224),Image.NEAREST)

        xml_path=os.path.join(self.xml_dir,f"{name}.xml")
        if os.path.exists(xml_path):
            tree=ET.parse(xml_path)
            root=tree.getroot()
            obj=root.find("object")
            bb=obj.find("bndbox")
            xmin=float(bb.find("xmin").text)
            ymin=float(bb.find("ymin").text)
            xmax=float(bb.find("xmax").text)
            ymax=float(bb.find("ymax").text)
            w=xmax-xmin
            h=ymax-ymin
            bbox=torch.tensor([xmin+w/2,ymin+h/2,w,h],dtype=torch.float32)
        else:
            bbox=torch.zeros(4,dtype=torch.float32)

        if self.transform:
            img=self.transform(img)

        mask=torch.tensor(np.array(mask),dtype=torch.long)-1
        return img,class_id,bbox,mask
