import os
import shutil
from tqdm import tqdm

# Define the source and destination paths
source_path = "/data/20230414/annual_meeting/am/"  # replace with your source image directory
destination_path = "/app/results_pipeline/clean_patches/"  # replace with your destination directory

# Ensure the destination path exists
os.makedirs(destination_path, exist_ok=True)

# Process each file in the source path
for filename in tqdm(os.listdir(source_path)):
    if filename.endswith(".jpg"):
        slide_id = filename[:12]
        slide_folder = os.path.join(destination_path, slide_id)
        
        # Create the slide folder if it doesn't exist
        os.makedirs(slide_folder, exist_ok=True)
        
        # Construct full file paths
        source_file = os.path.join(source_path, filename)
        destination_file = os.path.join(slide_folder, filename)
        
        # Move the file to the new folder
        shutil.copy(source_file, destination_file)

print("Images moved successfully.")
