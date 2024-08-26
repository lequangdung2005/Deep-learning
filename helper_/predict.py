import torch
import model_builder
from PIL import Image
from pathlib import Path
import torchvision
class_names = ["pizza", "steak", "sushi"]
device = "cuda" if torch.cuda.is_available() else "cpu"
def predict(img_path):
    img = torchvision.io.read_image(str(img_path)).type(torch.float32)
    img=img/255
    
    transform = torchvision.transforms.Resize(size=(64, 64))
    
    img = transform(img)
    
    img=img.to(device)

    
    
    model = model.to(device)
    model_save_path="models/05_going_modular_script_mode_tinyvgg_model.pth"
    model.load_state_dict(torch.load(model_save_path))
    
    model.to(device)
    
    model.eval()
    
    with torch.inference_mode():
        pred=model(img.unsqueeze(dim=0))
        
        label=class_names[pred.argmax()]
        
        print(label)
