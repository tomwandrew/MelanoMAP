import os
from openslide import OpenSlide
import re

def save_thumbnail(wsi_path, output_path, width=512):
    slide = OpenSlide(wsi_path)
    scale = width / slide.dimensions[0]
    height = int(slide.dimensions[1] * scale)
    thumbnail = slide.get_thumbnail((width, height))
    thumbnail.save(output_path)

def process_slides(data_dir):
    files = os.listdir(data_dir)
    slides = {}
    
    # Organize files by ID and Number, case-insensitive for staining
    for file in files:
        match = re.match(r'(\w+)_(am|lo)_(\d+)\.svs', file, re.IGNORECASE)
        if match:
            base_name = f"{match.group(1)}_{match.group(3)}"  # ID_Number
            staining = match.group(2).lower()  # Normalize staining to lowercase
            if base_name not in slides:
                slides[base_name] = []
            slides[base_name].append((staining, file))
    
    # Process each pair of slides with the same ID and Number
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_count = 0
    start_index = 10  # Start processing from the fifth pair
    current_pair = 0 # Current pair index

    for base_name, slide_files in slides.items():
        staining_types = {s[0] for s in slide_files}
        if len(slide_files) == 2 and 'am' in staining_types and 'lo' in staining_types:
            if current_pair >= start_index and processed_count < max_count:
                for staining, slide_file in slide_files:
                    wsi_path = os.path.join(data_dir, slide_file)
                    output_path = os.path.join(script_dir, f"{base_name}_{staining}_thumbnail.png")
                    save_thumbnail(wsi_path, output_path)
                processed_count += 1
            current_pair += 1
            if processed_count >= max_count:
                break

# Example usage
data_dir = "/data/histos2segment"
process_slides(data_dir)