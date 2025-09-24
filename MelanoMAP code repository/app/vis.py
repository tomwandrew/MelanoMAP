from PIL import Image
import numpy as np
import argparse
from glob import glob
import os
import pandas as pd

from skimage.io import imsave

folder_path = '/app/results_pipeline/masks'

# Paint numbers as colors
def paint_numbers(img, orig_img = None, weights = {0: 0.5, 1: 0.5}, colors = {0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1), 4: (1, 1, 1)}):
    result = np.zeros((*img.shape, 3), dtype=np.float32)
    for i in np.unique(img):
        result[img == i] = colors[i]

    # To gray scale (3 channels)
    if orig_img is not None:
        orig_img = np.stack([np.mean(orig_img, axis=2, keepdims=False)] * 3, axis=2)
        result = weights[0] * orig_img + weights[1] * result

    return result


def mask_visualization(df):
    path = df.iloc[2]['path']

    img = Image.open(path)
    orig_img = Image.open(path.replace('masks', 'patches').replace('/lo', '').replace('/am', '').replace('.png', '.jpg'))

    img = np.array(img)
    orig_img = np.array(orig_img)/255
    img = paint_numbers(img, orig_img)

    imsave('test.png', np.uint8(img*255))
    return img

def get_all_masks(folder_path):
    masks = glob(os.path.join(folder_path, '**', '*.png'), recursive=True)
    df = pd.DataFrame(masks, columns=['path'])

    df['staining'] = df['path'].apply(lambda x: x.split('/')[-3])
    df['slide_id'] = df['path'].apply(lambda x: x.split('/')[-2])
    df['image_id'] = df['path'].apply(lambda x: x.split('/')[-1])

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize a mask image.')
    parser.add_argument('--path', type=str, default=folder_path, help='Path to the mask image.')

    args = parser.parse_args()
    path = args.path

    df = get_all_masks(path)
    mask_visualization(df)
