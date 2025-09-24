import os
from glob import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm

from ai_functionalities.predict_tissue_background import predict_single_image

from skimage.io import imsave

# This is BGR
color_dict = {
    0: (0, 0, 0), # Background
    1: (255, 0, 0), # Tissue - Blue
    2: (0, 255, 0), # Tumor - Green
    3: (0, 0, 255), # Biomarker - Red
}

def label_image(image, mask):
    if mask is None:
        raise ValueError("Mask is None, cannot label image")

    mask_3 = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(len(color_dict)):
        mask_3[mask == i] = color_dict[i]

    labeled_image = cv2.addWeighted(image, 1, mask_3, 0.5, 0)
    return labeled_image

def compute_proportion_of_biomarkers(mask, biomarker_label = 3):
    return np.sum(mask == biomarker_label) / (mask.shape[0] * mask.shape[1])


def detect_darker_pixels(image, num_pixels=200):
    values_of_pixels = image.reshape(-1)
    
    # Sort pixels by value and get the 200 darkest
    darkest_pixels = np.sort(values_of_pixels)[:num_pixels]
    
    # Compute mean of the darkest pixels
    mean_darkest_pixels = np.mean(darkest_pixels)
    
    return mean_darkest_pixels


def label_am(image: str) -> None:
    image_name = os.path.basename(image)
    path = os.path.dirname(image)
        
    original_image = imread(image)
    mask_path = image.replace('.jpg', '.png').replace('patches', 'masks')

    mask = imread(mask_path, as_gray=True).astype(np.uint8)

    # Skip if mask doesnt have 3
    if 3 not in np.unique(mask):
        return

    labeled_image = label_image(original_image, mask)

    save_image = image.replace('patches', 'labeled')
    save_path = os.path.dirname(save_image)
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(save_image, labeled_image)

def label_lo(image: str) -> None:
    image_name = os.path.basename(image)
    path = os.path.dirname(image)
        
    original_image = imread(image)
    mask_path = image.replace('.jpg', '.png').replace('patches', 'masks')

    mask = imread(mask_path, as_gray=True).astype(np.uint8)

    # Skip if mask doesnt have 3
    if 3 not in np.unique(mask):
        return

    labeled_image = label_image(original_image, mask)

    save_image = image.replace('patches', 'labeled')
    save_path = os.path.dirname(save_image)
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(save_image, labeled_image)

def detect_staining(image_path: str) -> Optional[str]:
    if 'am'  in image_path.split('/')[-2].lower():
        return 'am'
    elif 'lo' in image_path.split('/')[-2].lower():
        return 'lo'
    else:
        return None

def label_masks(image_path: str) -> None:
    images = glob(f'results_pipeline/patches/**/*.jpg', recursive=True)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow
    
    for image in tqdm(images, desc="Labeling masks"):
        try:
            staining = detect_staining(image)

            if staining == 'am':
                label_am(image)
            elif staining == 'lo':
                label_lo(image)
            else:
                raise ValueError(f"Staining not recognized: {staining}")
        except Exception as e:
            continue



def main() -> None:
    label_masks("results_pipeline/patches/")

if __name__ == "__main__":
    main()