import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
from scipy.stats import ttest_ind


def compute_tumor(row, marker_mask):
    img = Image.open(row['img_path'])
    mask = Image.open(row['mask_path'])

    img = np.array(img)
    mask = np.array(mask)

    # Number for the tumor mask is always 2
    tumor = mask[mask == 2]#

    # Number for AM=2 and LO=3
    marker = mask[mask == 3]

    return np.sum(tumor), np.sum(marker)

def compute_image_stats(row, marker_mask):
    img = Image.open(row['img_path'])
    mask = Image.open(row['mask_path'])

    # Sum pixels where mask == 2
    mask = np.array(mask)
    img = np.array(img)

    # Transform img to hue
    img = rgb2hsv(img)

    # TODO: figure out why we are using only the value channel
    img = img[:, :, 2]
    marker = img[mask == marker_mask]

    hist = np.histogram(marker[:], bins=256, range=(0, 1))[0]
    # return mode
    # mode_value = mode(am, keepdims=False).mode
    return hist

def get_pixel_values(row, marker_mask):
    img_path = row['img_path']
    mask_path = row['mask_path']

    img = Image.open(img_path)
    mask = Image.open(mask_path)

    img = np.array(img)
    mask = np.array(mask)

    marker_masked = img[mask == marker_mask]

    return marker_masked

def compute_p_value_am(df):
    tumor = df[df['tumor'] == True]['mode'].values
    am = df[df['tumor'] == False]['mode'].values

    return ttest_ind(tumor, am)[1]