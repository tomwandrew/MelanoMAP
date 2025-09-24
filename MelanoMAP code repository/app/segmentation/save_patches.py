from tqdm import tqdm
import argparse
import os
 
import pandas as pd
import h5py
import openslide
from PIL import Image

def save_regions(h5_coords, seg_level, histo_name):
    save_path = '/app/results_pipeline/patches/'
    images_path = '/data/histos2segment/'
    try:
        wsi = openslide.OpenSlide(images_path + histo_name + '.svs')
        # TODO: if we can't find mpp_x we should do something about it
        mpp_x = wsi.properties[openslide.PROPERTY_NAME_MPP_X]  # Get microns per pixel in x dimension

        # Create a subdirectory for the current WSI
        wsi_save_path = os.path.join(save_path, histo_name)
        os.makedirs(wsi_save_path, exist_ok=True)
        print('Saving patches')
        for x_coord, y_coord in tqdm(h5_coords, desc='Saving WSI patches...'):
            # Update patch_path to include the subdirectory

            save_path = f'{histo_name} [d=5,x={x_coord},y={y_coord},w=1280,h=1280,mpp={mpp_x}].jpg'
            patch_path = os.path.join(wsi_save_path, save_path)
           
            # Check if the file already exists
            if not os.path.exists(patch_path):
                patch = wsi.read_region((x_coord, y_coord), seg_level, (1280,1280)).convert('RGB')
                # CommentToMarc: This is where we make the downsampling of the patch from 1280 to 256
                # We use bilinear for resampling as its the best trade-off between quality and speed
                resized_patch = patch.resize((256, 256), resample=Image.BILINEAR) 

                patch_mean = sum(resized_patch.convert("L").getdata()) / (256 * 256)
                if patch_mean > 245:
                    continue
                else:
                    resized_patch.save(patch_path)

    except Exception as e:
        print(f'Error processing {histo_name}: {e}')
        return []  # Return an empty list to indicate failure
    return 'ok'

def patch_wsi(histo_name, h5_dir):
    with h5py.File(os.path.join(h5_dir, histo_name + '.h5'), 'r') as f:
        h5_coords = f['coords']
        seg_level = h5_coords.attrs['patch_level']

        return save_regions(h5_coords, seg_level, histo_name)

def save_patches(h5_dir):

    file_names = [os.path.splitext(x)[0] for x in os.listdir(h5_dir) if x.endswith('.h5')]
    all_patches = []  # List to store all patch information


    for histo_name in file_names:
        # Update the description for each histo_name
    #    pbar.set_description(f'Patching slide {histo_name}')
        try:
            all_patches.extend(patch_wsi(histo_name, h5_dir))  # Accumulate patch information
        except Exception as e:
            print(f'Filename {histo_name} could not be processed. Error: {e}')
    #    pbar.update(1)  # Manually update the progress bar

    #pbar.close()


if __name__=='__main__':

    files_path =  '/home/carlos.hernandez/PhD/CLAM/tcga_slides/patches/'
    file_names = [os.path.splitext(x)[0] for x in os.listdir(files_path) if x.endswith('.h5')]
