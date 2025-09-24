import cv2
import numpy as np
from glob import glob
import os

from tqdm import tqdm
import argparse
from PIL import Image


from tqdm import tqdm

def label2rgb(mask, image, bg_label=0):
    color_dict = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
        4: (255, 255, 0),
        5: (255, 0, 255),
        6: (0, 255, 255),
        7: (255, 255, 255),
    }

    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label in np.unique(mask):
        if label == bg_label:
            continue
        mask_rgb[mask == label] = color_dict[label]

    # Unique colors
    unique_colors = np.unique(mask_rgb.reshape(-1, mask_rgb.shape[2]), axis=0)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    image_gray = np.array([image_gray, image_gray, image_gray])
    image_gray = np.transpose(image_gray, (1, 2, 0))

    mask_rgb = cv2.addWeighted(mask_rgb, 0.5, image_gray, 0.5, 0)
    # mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)

    return mask_rgb


def join_masks(image_path, mask_path, bg_path):
    image = cv2.imread(image_path)
    mask = Image.open(mask_path)
    mask = np.array(mask)
    bg = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)

    # Create the result mask based on the conditions
    res_mask = mask.copy()
    res_mask[(mask == 0) & (bg == 1)] = 5  # background

    # Replace values, 5 -> 0, 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 3 in a new image
    res_mask_new = res_mask.copy()
    res_mask_new[res_mask == 5] = 0 # BG
    res_mask_new[res_mask == 0] = 1 # TISSUE
    res_mask_new[res_mask == 1] = 2 # Tumour
    res_mask_new[res_mask == 2] = 3 # Biomarker 1
    res_mask_new[res_mask == 3] = 3 # Biomarker 2

    res_mask_new_label = label2rgb(res_mask_new, image, bg_label=0)
    
    return res_mask_new, res_mask_new_label

# Example usage
if __name__ == "__main__":
    split = 'train'
    biomarker = 'lo'

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default=f'/data/20230414/processed/{biomarker}/{split}/images')
    parser.add_argument('--mask_path', type=str, default=f'/data/20230414/processed/{biomarker}/{split}/masks')
    parser.add_argument('--bg_path', type=str, default=f'/data/20230414/processed/{biomarker}/{split}/bg')

    parser.add_argument('--dst_folder', type=str, default=f'/data/20230414/processed/{biomarker}/{split}/joined')
    parser.add_argument('--label_folder', type=str, default=f'/data/20230414/processed/{biomarker}/{split}/joined_label')

    args = parser.parse_args()


    os.makedirs(args.dst_folder, exist_ok=True)
    os.makedirs(args.label_folder, exist_ok=True)

    files = glob(os.path.join(args.folder_path, "*.jpg"))

    for image_path in tqdm(files):
        mask_path = os.path.join(args.mask_path, os.path.basename(image_path).replace(".jpg", ".png"))
        bg_path = os.path.join(args.bg_path, os.path.basename(image_path).replace(".jpg", ".png"))
        res_mask_new, res_mask_new_label = join_masks(image_path, mask_path, bg_path)
        cv2.imwrite(os.path.join(args.dst_folder, os.path.basename(image_path).replace(".jpg", ".png")), res_mask_new)
        cv2.imwrite(os.path.join(args.label_folder, os.path.basename(image_path).replace(".jpg", "_label.jpg")), res_mask_new_label)
