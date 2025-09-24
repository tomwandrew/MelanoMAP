from glob import glob
import pandas as pd
import os 
import argparse
import shutil
from tqdm import tqdm

def list_patches(image_path, staining, label, output_path):
    print(f"Listing patches for {label} with staining '{staining}' from {image_path}...")
    patches = glob(os.path.join(image_path, '**', '*.jpg'), recursive=True)
    patches = pd.DataFrame(patches, columns=['path'])
    patches['patch_name'] = patches['path'].apply(lambda x: x.split('/')[-1].split('.')[0])
    patches['slide_name'] = patches['path'].apply(lambda x: x.split('/')[-2])
    patches['staining'] = patches['slide_name'].apply(lambda x: x.split('_')[-2]).str.lower()
    patches = patches[patches['staining'] == staining]

    patches['label'] = label

    encode_path = {
        'negative': 'Background',
        'positive': 'Tissue'
    }
    patches['output_path'] = output_path + '/' + encode_path[label]
    
    print(f"Found {len(patches)} patches for {label}.")
    return patches


def copy_patches(patches):
    os.makedirs(patches['output_path'].iloc[0], exist_ok=True)
    
    print("Starting to copy patches...")
    for index, row in tqdm(patches.iterrows(), total=patches.shape[0], desc="Copying patches"):
        shutil.copy(row['path'], os.path.join(row['output_path'], row['patch_name'] + '.jpg'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path_negative', type=str, help='path to the image', default='/app/results_pre_august/patches')
    parser.add_argument('--image_path_positive', type=str, help='path to the image', default='/app/results_pre_august/clean_patches')

    parser.add_argument('--staining', type=str, help='staining', default='am')
    parser.add_argument('--output_path', type=str, help='output path', default='/data/tissue_n_background_marc/am')
    args = parser.parse_args()

    print("Processing negative patches...")
    patches_negative = list_patches(args.image_path_negative, args.staining, 'negative', os.path.join(args.output_path, 'train'))
    print("Processing positive patches...")
    patches_positive = list_patches(args.image_path_positive, args.staining, 'positive', os.path.join(args.output_path, 'train'))
    
    # Sample negative patches to match the number of positive patches
    # patches_negative = patches_negative.sample(n=len(patches_positive))

    # Create validation sets
    print("Creating validation sets...")
    validation_negative = patches_negative.sample(frac=0.2, random_state=42)
    validation_negative['output_path'] = os.path.join(args.output_path, 'validation', 'Background')
    validation_positive = patches_positive.sample(frac=0.2, random_state=42)
    validation_positive['output_path'] = os.path.join(args.output_path, 'validation', 'Tissue')

    # Copy patches for training
    copy_patches(patches_negative.drop(validation_negative.index))
    copy_patches(patches_positive.drop(validation_positive.index))

    # Copy validation patches
    copy_patches(validation_negative)
    copy_patches(validation_positive)


if __name__ == '__main__':
    main()
