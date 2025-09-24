import os
import shutil

import pandas as pd
import seaborn as sns
from typing import Tuple

from scipy.spatial import distance

from PIL import Image
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from glob import glob
from skimage import measure


from utils.handcraft_utils import compute_tumor


sns.set()

# python handcraft_lo.py
def safe_find_contours(blob):
    # Check if the blob's image is at least 2x2 in size
    if blob.image.shape[0] >= 2 and blob.image.shape[1] >= 2: 
        # If so, attempt to find contours
        return measure.find_contours(blob.image, level=0.5)
    else:
        # If the blob is too small, return an empty list
        return []

# CommentToMarc: This function computes the maximum gap between two blobs in the mask
# This needs improving, is currently one of the weak points.
def compute_spatial_blob_gaps(row, percentage_of_image):
    # Load the image
    mask = Image.open(row['mask_path'])
    # Convert the image to a numpy array
    mask = np.array(mask)
    # Determine the threshold based on a percentage of the image size
    max_distance_threshold = min(mask.shape[:2]) * percentage_of_image
    
    # Binarize the mask
    processed_mask = mask == 3
    # Label the connected regions in the mask
    labeled_blobs = measure.label(processed_mask, background=0)
    # Calculate properties of labeled regions (blobs)
    blobs = measure.regionprops(labeled_blobs)

    # Calculate the centroids for each blob
    centroids = np.array([blob.centroid for blob in blobs])
    
    # Initialize an array to hold the minimum distances
    min_distances = np.full(len(blobs), np.inf)
    
    # Compute the centroid distances to find neighbors within the threshold
    if len(blobs) > 1:
        centroid_distances = distance.cdist(centroids, centroids)
        np.fill_diagonal(centroid_distances, np.inf)
        
        # Mask out distances that are above the threshold
        centroid_distances[centroid_distances > max_distance_threshold] = np.inf
        
        for i, blob_i in enumerate(blobs):
            distances_to_others = centroid_distances[i]
            nearest_neighbor_within_threshold = np.argmin(distances_to_others)
            
            if distances_to_others[nearest_neighbor_within_threshold] < np.inf:
                blob_j = blobs[nearest_neighbor_within_threshold]
                
                # Retrieve contours for blob_i and blob_j using the safe function
                contours_i = safe_find_contours(blob_i)
                contours_j = safe_find_contours(blob_j)
                
                if contours_i and contours_j:
                    # Adjust contours to the top-left of the bounding box
                    adjusted_contour_i = contours_i[0] + np.array([blob_i.bbox[0], blob_i.bbox[1]])
                    adjusted_contour_j = contours_j[0] + np.array([blob_j.bbox[0], blob_j.bbox[1]])
                    
                    # Compute the minimum distance between two sets of points (contours)
                    min_distances[i] = min(distance.euclidean(point_i, point_j)
                                           for point_i in adjusted_contour_i
                                           for point_j in adjusted_contour_j)
    
    # Filter out the infinite values which indicate no neighbor within the threshold
    min_distances = min_distances[min_distances < np.inf]

    # Find the maximum gap
    max_gap = np.max(min_distances) if min_distances.size > 0 else 0
    return max_gap

def hc_lo_detection(image_path: str) -> Tuple[pd.DataFrame, str, str]:
    # Read all lo data
    files = glob(image_path + '/**/*.jpg', recursive=True)
  #  import pdb; pdb.set_trace()
    files = pd.DataFrame(files, columns=['img_path'])
    # Extract image name and mask path from img filename
    files['image_name'] = files['img_path'].str.extract(r'([^\/]+)\.jpg$')[0]
    files['slide'] = files['img_path'].str.split('/').str[-2]
    files['mask_path'] = 'results_pipeline/masks/lo/' + files['slide'] + '/' + files['image_name'] + '.png'


    files['details'] = files['image_name'].str.split(' ').str[1]
    files['slide'] = files['image_name'].str.split(' ').str[0]

    files['downsample'] = files['details'].str.split(',').str[0].str.split('=').str[1].map(float)
    files['mpp'] = files['details'].str.split(',').str[5].str.split('=').str[1].str[:-1].map(float)

    # Check for the presence of tumour and lo (lo==3) in the predicted mask
    print('Computing tumor and lo...')
    files['tumor'], files['lo'] = zip(*files.apply(compute_tumor, args=(3,), axis=1))
    files['tumor']  = files['tumor'] > 0
    files['lo']  = files['lo'] > 0

    # Filter files without lo in them
    files = files[files['lo'] == True]
    # files = files[files['tumor'] == True]

    # CommentToMarc:  Which percentage of the image to consider to be too far from one glob to the other
    # I tried watershading + reconstruct but did not obtain good results
    percentage_of_image_threshold = 0.3
    # Compute the number of blobs in the mask#
    if len(files) == 0:
        return files, "No patches with LO founds", "Not applicable"
    files['max_distance'] = files.apply(lambda row: compute_spatial_blob_gaps(row, percentage_of_image_threshold), axis=1)
   # We multiply the max distance obtained with the MPP and we have to scale it due to the downsample
    files['positive_result'] = (files['max_distance'] * files['mpp'] * files['downsample']) > 20

    # Take only the paths with positive gap result
    files_hc_positive = files[files['positive_result']==True].copy()

    # Create a directory to save positive images that were deemed positive by the heuristic
    for slide_id in files_hc_positive['slide'].unique():
        folder_path = f'/app/results_pipeline/handcraft/lo_positive/{slide_id}'
        os.makedirs(folder_path, exist_ok=True)

    # Copy positive images to respective folders
    for index, row in files_hc_positive.iterrows():
        image_path = row['img_path']
        slide_id = row['slide']
        destination_path = f'/app/results_pipeline/handcraft/lo_positive/{slide_id}/{os.path.basename(image_path)}'
        shutil.copy(image_path, destination_path)

    overall_prediction = 'Lost expression' if len(files_hc_positive) > 0 else 'Maintained expression'
    return files, overall_prediction, 'Not applicable'
    # python handcraft_lo.py
