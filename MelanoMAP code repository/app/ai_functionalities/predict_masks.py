import numpy as np
import argparse
from tqdm import tqdm
from stqdm import stqdm

from glob import glob
from PIL import Image
import sys
import torch
import pandas as pd
import segmentation_models_pytorch as smp
import os

from utils.utils import *
import warnings

warnings.filterwarnings("ignore", message=".*Implicit dimension choice for softmax has been deprecated.*", category=UserWarning)

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, files, preprocess_input):
        self.files = files
        self.preprocess_input = preprocess_input

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        row = self.files.iloc[idx]
        image_path = row['path']
        slide_id = row['slide']

        image = Image.open(image_path)
        image = np.array(image)
        image = self.preprocess_input(image).transpose(2, 0, 1)
        image_name = row['image']

        return image, slide_id, image_name

def ai_segment(image_path, staining='default'):
    # python predict_folder.py
    # LO model

    checkpoint_path = f"./models/best_model_{staining}_4.pth"
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(checkpoint_path, map_location=DEVICE)

    preprocess_input = smp.encoders.get_preprocessing_fn('efficientnet-b0')
    images = glob(image_path + "/*.jpg", recursive=True)
    files = pd.DataFrame({"path": images})
    files['image'] = files['path'].str.split('/').str[-1].str[:-4]
    files['details'] = files['image'].str.split(' ').str[1]
    files['slide'] = files['image'].str.split(' ').str[0]

    files['downsample'] = files['details'].str.split(',').str[0].str.split('=').str[1].map(float)
    
    files['x'] = files['details'].str.split(',').str[1].str.split('=').str[1].map(float)
    files['y'] = files['details'].str.split(',').str[2].str.split('=').str[1].map(float)
    files['width'] = files['details'].str.split(',').str[3].str.split('=').str[1].map(float)
    files['height'] = files['details'].str.split(',').str[4].str.split('=').str[1].str[:-1].map(float)
    files['mpp'] = files['details'].str.split(',').str[5].str.split('=').str[1].str[:-1].map(float)

    # Create DataLoader
    dataset = CustomDataset(files, preprocess_input)

    ## TODO: Add more num_workers once I am not downloading any more WSI
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True, num_workers=2) #, num_workers=1

    original_images = []
    predicted_masks = []
    slide_ids_list = []
    image_names_list = []
    
    for images, slide_ids, image_names in stqdm(dataloader, desc='Obtaining masks...', file=sys.stdout):
        images = images.to(DEVICE).float()
        pr_masks = model.predict(images)

        pr_masks = np.argmax(pr_masks.cpu().numpy(), axis=1)

        for i in range(len(pr_masks)):
            if len(original_images) < 4 and (1 in pr_masks[i] and (2 in pr_masks[i] or 3 in pr_masks[i])):
                original_image_path = files.loc[files['image'] == image_names[i], 'path'].values[0] # type: ignore
                original_image = np.array(Image.open(original_image_path))

                original_images.append(original_image)
                binary_mask_1 = np.where(pr_masks[i] == 1, 255, 0).astype(np.uint8)
                binary_mask_2 = np.where((pr_masks[i] == 2) | (pr_masks[i] == 3), 255, 0).astype(np.uint8)
                predicted_masks.append((binary_mask_1, binary_mask_2))
                slide_ids_list.append(slide_ids[i])
                image_names_list.append(image_names[i])

            # Save the mask
            pr_mask = pr_masks[i].astype(np.uint8)
            pr_mask = Image.fromarray(pr_mask)
            save_dir = f'results_pipeline/masks/{staining}/{slide_ids[i]}/'
            if staining == "he":
            save_dir = f'results_pipeline/masks/HnE/{slide_ids[i]}/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            pr_mask.save(os.path.join(save_dir, f'{image_names[i]}.png'))
    data = {
        'original_image': original_images,
        'tumour_mask': [mask[0] for mask in predicted_masks],
        'staining mask': [mask[1] for mask in predicted_masks],
        'slide_id': slide_ids_list,
        'image_name': image_names_list
    }
    result_df = pd.DataFrame(data)
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="UNET AMBLor Test",
    )
    parser.add_argument(
        "--backbone", type=str, help="backbone", default="efficientnet-b0"
    )
    parser.add_argument(
        "--classes",
        type=str,
        help="classes",
        default=["background", "tumour", "am", "lo"],
    )
    parser.add_argument('--staining', type=str, choices=['lo', 'am', 'he'], default='lo')
    args = parser.parse_args()

    image_dirs = glob(f'/data/20230414/{args.staining}/*/')


    print(f"Input arguments: backbone={args.backbone}, classes={args.classes}, staining={args.staining}")
    for image_dir in image_dirs:
        args.image_dir = image_dir
        ai_segment(args)
