import os
from glob import glob
from typing import List, Tuple, Optional
from collections import defaultdict

import cv2
import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm

from shutil import copyfile

from ai_functionalities.predict_tissue_background import predict_single_image


# Utility functions
def detect_staining(image_path: str) -> Optional[str]:
    if 'am' in image_path.split('/')[-2].lower():
        return 'am'
    elif 'lo' in image_path.split('/')[-2].lower():
        return 'lo'
    else:
        return None

def compute_proportion_of_biomarkers(mask, biomarker_label=3):
    return np.sum(mask == biomarker_label) / (mask.shape[0] * mask.shape[1])

def detect_darker_pixels(image, num_pixels=200):
    values_of_pixels = image.reshape(-1)
    darkest_pixels = np.sort(values_of_pixels)[:num_pixels]
    return np.mean(darkest_pixels)


def process_am_staining(image_path: str, mask: np.ndarray, original_image: np.ndarray) -> bool:
    # Check if the image is mostly white
    if is_mostly_white(original_image):
        print(f"Image is mostly white, discarding AM staining: {image_path}")
        return False

    # Check if the biomarker region is too dark
    if is_biomarker_region_too_dark(original_image, mask):
        print(f"Biomarker region is too dark, discarding AM staining: {image_path}")
        return False

    # Check the proportion of tissue or tumor
    tissue_tumor_proportion = compute_proportion_of_tissue_or_tumor(mask)
    if tissue_tumor_proportion < 0.1:  # Adjust this threshold as needed
        print(f"Insufficient tissue or tumor proportion ({tissue_tumor_proportion:.2f}) for AM staining: {image_path}")
        return False
    
    # Compute proportion of biomarkers
    biomarker_proportion = compute_proportion_of_biomarkers(mask)
    
    # Check if the biomarker proportion is too low
    if biomarker_proportion < 0.02:
        return False
    
    # Detect darker pixels in the original image
    darkness = detect_darker_pixels(original_image)
    
    # Check if the image is too light (adjust threshold as needed)
    if darkness > 200:
        return False
    
    # Add more criteria for AM staining as needed
    
    return True

def process_lo_staining(image_path: str, mask: np.ndarray, original_image: np.ndarray) -> bool:
    # Check if the image is mostly white
    if is_mostly_white(original_image):
        print(f"Image is mostly white, discarding AM staining: {image_path}")
        return False

    # Check if the biomarker region is too dark
    if is_biomarker_region_too_dark(original_image, mask):
        print(f"Biomarker region is too dark, discarding AM staining: {image_path}")
        return False

    # Check the proportion of tissue or tumor
    tissue_tumor_proportion = compute_proportion_of_tissue_or_tumor(mask)
    if tissue_tumor_proportion < 0.1:  # Adjust this threshold as needed
        print(f"Insufficient tissue or tumor proportion ({tissue_tumor_proportion:.2f}) for AM staining: {image_path}")
        return False
    
    # Compute proportion of biomarkers
    biomarker_proportion = compute_proportion_of_biomarkers(mask)
    
    # Check if the biomarker proportion is too low
    if biomarker_proportion < 0.02:
        return False
    
    # Detect darker pixels in the original image
    darkness = detect_darker_pixels(original_image)
    
    # Check if the image is too light (adjust threshold as needed)
    if darkness > 200:
        return False

    return True

def is_mostly_white(image: np.ndarray, threshold: float = 0.95) -> bool:
    """
    Check if the image is mostly white (or very light-colored).
    
    :param image: The input image as a numpy array.
    :param threshold: The threshold for considering an image mostly white (default: 95% of pixels).
    :return: True if the image is mostly white, False otherwise.
    """
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Count pixels that are very light (close to white)
    white_pixels = np.sum(gray > 240)
    total_pixels = gray.size
    
    return bool(white_pixels / total_pixels > threshold)

def compute_proportion_of_tissue_or_tumor(mask, tissue_label=1, tumor_label=2):
    return np.sum((mask == tissue_label) | (mask == tumor_label)) / (mask.shape[0] * mask.shape[1])

def is_biomarker_region_too_dark(original_image: np.ndarray, mask: np.ndarray, darkness_threshold: int = 50, area_threshold: float = 0.5) -> bool:
    """
    Check if a large portion of the biomarker region is too dark.
    
    :param original_image: The original image as a numpy array.
    :param mask: The mask image as a numpy array.
    :param darkness_threshold: Pixel value below which a pixel is considered dark.
    :param area_threshold: Proportion of dark pixels in biomarker region to consider it too dark.
    :return: True if the biomarker region is too dark, False otherwise.
    """
    # Convert to grayscale if it's a color image
    if len(original_image.shape) == 3:
        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = original_image
    
    # Create a binary mask for biomarker regions
    biomarker_mask = (mask == 3)
    
    # Apply the biomarker mask to the grayscale image
    biomarker_region = gray[biomarker_mask]
    
    # Count dark pixels in the biomarker region
    dark_pixels = np.sum(biomarker_region < darkness_threshold)
    total_pixels = biomarker_region.size
    
    # Calculate the proportion of dark pixels
    dark_proportion = dark_pixels / total_pixels if total_pixels > 0 else 0
    
    return bool(dark_proportion > area_threshold)

def process_and_save_image(image_path: str) -> None:
    original_image = imread(image_path)
    
    # Check if the image is mostly white
    if is_mostly_white(original_image):
        print(f"Image is mostly white, discarding: {image_path}")
        return
    
    staining = detect_staining(image_path)
    mask_path = image_path.replace('.jpg', '.png').replace('patches', 'masks')
    mask = imread(mask_path, as_gray=True).astype(np.uint8)

    if 3 not in np.unique(mask):
        print(f"No biomarker found in {image_path}")
        return

    if 1 not in np.unique(mask):
        print(f"No tissue found in {image_path}")
        return

    labeled_image_path = image_path.replace('patches', 'labeled')
    
    if staining == 'am':
        if process_am_staining(image_path, mask, original_image):
            save_image = image_path.replace('patches', 'labeled_clean')
            save_path = os.path.dirname(save_image)
            os.makedirs(save_path, exist_ok=True)
            copyfile(labeled_image_path, save_image)
        else:
            print(f"AM staining patch discarded: {image_path}")
    elif staining == 'lo':
        if process_lo_staining(image_path, mask, original_image):
            save_image = image_path.replace('patches', 'labeled_clean')
            save_path = os.path.dirname(save_image)
            os.makedirs(save_path, exist_ok=True)
            copyfile(labeled_image_path, save_image)
        else:
            print(f"LO staining patch discarded: {image_path}")
    else:
        print(f"Unknown staining type for {image_path}")

def label_staining(image_path: str) -> None:
    process_and_save_image(image_path)

# Main processing function
def label_masks(base_path: str) -> None:
    # Group images by folder
    folder_images = defaultdict(list)
    for image in glob(f'{base_path}**/*.jpg', recursive=True):
        folder = os.path.dirname(image)
        folder_images[folder].append(image)
    
    # Sort folders
    sorted_folders = sorted(folder_images.keys())
    if not sorted_folders:
        print("No folders with images found.")
        return
    
    for folder in sorted_folders:
        # Check if the folder contains AM staining images
        if 'lo' in folder.lower():
            print(f"Processing LO staining folder: {folder}")
            for image in tqdm(folder_images[folder], desc=f"Labeling masks in {os.path.basename(folder)}"):
                try:
                    label_staining(image)
                except Exception as e:
                    print(f"Error processing {image}: {e}")
                    continue  # Continue with the next image
        
        else:
            print(f"Skipping non-LO staining folder: {folder}")

def main() -> None:
    base_path = "results_pipeline/patches/"
    label_masks(base_path)

if __name__ == "__main__":
    main()