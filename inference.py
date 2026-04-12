import torch
from torchvision import transforms
from PIL import Image
from multitask import MultiTaskPerceptionModel
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
def predict(image_path:str): #inference script used for testing the models.
    model=MultiTaskPerceptionModel().to(DEVICE)
    model.eval()
    img=Image.open(image_path).convert("RGB")
    x=transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out=model(x)
    return out
    
if __name__=="__main__":
    result=predict("data/images/Abyssinian_1.jpg")
    print("classification:",result["classification"].argmax(1))
    print("localization:",result["localization"])
    print("segmentation:",result["segmentation"].argmax(1).shape)
