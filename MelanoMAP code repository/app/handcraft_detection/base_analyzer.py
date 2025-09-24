"""Base class for histological image analysis."""
import os
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod

class BaseAnalyzer(ABC):
    def __init__(self, output_dir='results_pipeline'):
        self.output_dir = output_dir
        self.base_dir = None  # To be set by child classes
        
    def _create_directories(self, *paths):
        """Create necessary output directories."""
        for path in paths:
            os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def load_image_and_mask(img_path, mask_path):
        """Load and convert image and mask to numpy arrays."""
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        return img, mask
    
    @abstractmethod
    def analyze_patch(self, img_path, mask_path):
        """Analyze a single patch."""
        pass
    
    @abstractmethod
    def analyze_slide(self, patch_data):
        """Analyze an entire slide."""
        pass
    
    @abstractmethod
    def plot_results(self, patch_stats, analysis_results, slide_name):
        """Plot analysis results."""
        pass 