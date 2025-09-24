import cv2
import numpy as np
from glob import glob
import os

from tqdm import tqdm
import argparse
from skimage.morphology import remove_small_holes, remove_small_objects, dilation, erosion # Add this import

def detect_background(image_path, threshold=230, min_blob_size=500):
    """
    Detect if the background of the image is close to white and label the pixels.

    :param image_path: Path to the image file.
    :param threshold: Threshold value to consider a color as white (0-255).
    :param min_blob_size: Minimum size of blobs to be considered as background.
    :return: A mask indicating background pixels.
    """
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a mask where pixels above the threshold are considered background
    background_mask = gray_image >= threshold
    
    # Remove small blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(background_mask.astype(np.uint8), connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_blob_size:
            background_mask[labels == i] = 0
    
    # Fill holes in the background mask
    background_mask = remove_small_holes(background_mask).astype(np.uint8)

    for _ in range(5):
        background_mask = dilation(background_mask).astype(np.uint8)

    for _ in range(5):
        background_mask = erosion(background_mask).astype(np.uint8)
    
    return background_mask

# Example usage
if __name__ == "__main__":
    split = 'validation'
    biomarker = 'lo'

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default=f'/data/20230414/processed/{biomarker}/{split}/images')
    parser.add_argument('--dst_folder', type=str, default=f'/data/20230414/processed/{biomarker}/{split}/bg')
    args = parser.parse_args()
    folder_path = args.folder_path
    dst_folder = args.dst_folder

    os.makedirs(dst_folder, exist_ok=True)

    for image_path in tqdm(glob(os.path.join(folder_path, "*.jpg"))):
        background_mask = detect_background(image_path)
        # Save to PNG to that I can see
        cv2.imwrite(f'{dst_folder}/{os.path.basename(image_path).replace(".jpg", ".png")}', background_mask)