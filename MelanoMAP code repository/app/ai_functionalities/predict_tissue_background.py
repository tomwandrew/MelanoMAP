import os
import numpy as np
import torch
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from stqdm import stqdm

import numpy as np
import os
import torch
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [
            os.path.join(image_dir, img) for img in os.listdir(image_dir)
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        if self.transform:
            img = self.transform(image=img)['image']

        return img, img_path

def load_model(staining):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = models.efficientnet_v2_s()
    num_ftrs = model.classifier[1].in_features
    
    model.classifier[1] = torch.nn.Linear(num_ftrs, 2)

    if staining == 'both':
        model_path = f'models/best_classifier_background_tissue.pth'
    else:
        model_path = f'models/best_classifier_background_tissue_{staining}.pth'
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
     
    model.to(device)
    model.eval()
    
    return model

def get_inference_augmentation():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def softmax(outputs):
    return torch.nn.functional.softmax(outputs, dim=1)

def ai_filter(image_path, staining = 'both'):
    # Load model
    model = load_model(staining)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create DataLoader
    dataset = CustomImageDataset(image_path, transform=get_inference_augmentation())
    dataloader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=2)

    for images, paths in stqdm(dataloader, desc='Filtering tissue from background with AI model'):

        images = images.to(device)
        outputs = model(images)
        outputs = softmax(outputs)
        preds = outputs.cpu().detach().numpy()[:, 1] > 0.5

        for i in range(images.size(0)):
            if preds[i] == False:
                return False # Discard image
                
        return True


def predict_single_image(image_path, staining):
    model = load_model(staining)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load and transform the single image
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)
    transform = get_inference_augmentation()
    img = transform(image=img)['image']
    img = img.unsqueeze(0)  # Add batch dimension

    img = img.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(img)
        output = softmax(output)
        pred = output.cpu().numpy()[0, 1] > 0.5

    return pred, output.cpu().numpy()[0, 1]
