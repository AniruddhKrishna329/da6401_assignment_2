import os
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms
from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
EPOCHS=30
BATCH_SIZE=16
LR=1e-4
CHECKPOINT_DIR="checkpoints"
os.makedirs(CHECKPOINT_DIR,exist_ok=True)

transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

def get_loaders(split="trainval"):
    dataset=OxfordIIITPetDataset(root="data",split=split,transform=transform)
    return DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)

def train_classifier():
    wandb.init(project="da6401_assignment_2",name="classifier",reinit=True)
    loader=get_loaders()
    model=VGG11Classifier().to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    criterion=nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        total_loss,correct,total=0,0,0
        for img,label,_,_ in loader:
            img,label=img.to(DEVICE),label.to(DEVICE)
            opt.zero_grad()
            out=model(img)
            loss=criterion(out,label)
            loss.backward()
            opt.step()
            total_loss+=loss.item()
            correct+=(out.argmax(1)==label).sum().item()
            total+=label.size(0)
        wandb.log({"cls/loss":total_loss/len(loader),"cls/acc":correct/total},step=epoch+1)
        print(f"Epoch {epoch+1} cls loss:{total_loss/len(loader):.4f} acc:{correct/total:.4f}")
    torch.save(model.state_dict(),os.path.join(CHECKPOINT_DIR,"classifier.pth"))
    wandb.finish()

def train_localizer():
    wandb.init(project="da6401_assignment_2",name="localizer_v2",reinit=True)
    loader=get_loaders()
    model=VGG11Localizer().to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma=0.5)
    mse=nn.MSELoss()
    iou=IoULoss()
    for epoch in range(50):
        model.train()
        total_loss=0
        for img,_,bbox,_ in loader:
            img,bbox=img.to(DEVICE),bbox.to(DEVICE)
            opt.zero_grad()
            out=model(img)
            loss=mse(out,bbox)+iou(out,bbox)
            loss.backward()
            opt.step()
            total_loss+=loss.item()
        scheduler.step()
        wandb.log({"loc/loss":total_loss/len(loader)},step=epoch+1)
        print(f"Epoch {epoch+1} loc loss:{total_loss/len(loader):.4f}")
    torch.save(model.state_dict(),os.path.join(CHECKPOINT_DIR,"localizer.pth"))
    wandb.finish()

def train_unet():
    wandb.init(project="da6401_assignment_2",name="unet",reinit=True)
    loader=get_loaders()
    model=VGG11UNet().to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    criterion=nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        total_loss=0
        for img,_,_,mask in loader:
            img,mask=img.to(DEVICE),mask.to(DEVICE)
            opt.zero_grad()
            out=model(img)
            loss=criterion(out,mask)
            loss.backward()
            opt.step()
            total_loss+=loss.item()
        wandb.log({"seg/loss":total_loss/len(loader)},step=epoch+1)
        print(f"Epoch {epoch+1} seg loss:{total_loss/len(loader):.4f}")
    torch.save(model.state_dict(),os.path.join(CHECKPOINT_DIR,"unet.pth"))
    wandb.finish()

if __name__=="__main__":
    train_classifier()
    train_localizer()
    train_unet()
