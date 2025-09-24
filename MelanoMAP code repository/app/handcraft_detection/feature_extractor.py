from skimage.feature import graycomatrix, graycoprops
from skimage import measure
from scipy import ndimage, stats
import numpy as np

class FeatureExtractor:
    """Class to extract various image features from biomarker regions."""
    
    @staticmethod
    def compute_texture_features(image_patch, biomarker_mask):
        """Compute GLCM texture features only in biomarker-positive regions.
        
        Args:
            image_patch: 2D numpy array of image intensities
            biomarker_mask: Boolean mask indicating biomarker presence
            
        Returns:
            dict: Dictionary of texture features
        """
        # Only analyze biomarker regions
        masked_image = image_patch.copy()
        masked_image[~biomarker_mask] = 0
        
        # Normalize and convert to uint8 for GLCM
        masked_image = ((masked_image - masked_image[biomarker_mask].min()) * 255 / 
                       (masked_image[biomarker_mask].max() - masked_image[biomarker_mask].min())).astype(np.uint8)
        
        # Compute GLCM only for biomarker regions
        glcm = graycomatrix(masked_image, distances=[1], angles=[0], 
                           levels=256, symmetric=True, normed=True)
        
        features = {
            'texture_contrast': graycoprops(glcm, 'contrast')[0, 0],
            'texture_dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],
            'texture_homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
            'texture_energy': graycoprops(glcm, 'energy')[0, 0],
            'texture_correlation': graycoprops(glcm, 'correlation')[0, 0],
            'texture_ASM': graycoprops(glcm, 'ASM')[0, 0]
        }
        return features
    
    @staticmethod
    def compute_morphological_features(biomarker_mask):
        """Compute shape and distribution features of biomarker regions.
        
        Args:
            biomarker_mask: Boolean mask indicating biomarker presence
            
        Returns:
            dict: Dictionary of morphological features
        """
        labeled_regions = measure.label(biomarker_mask)
        region_props = measure.regionprops(labeled_regions)
        
        if not region_props:
            return {
                'morph_num_regions': 0,
                'morph_mean_area': 0,
                'morph_mean_perimeter': 0,
                'morph_mean_eccentricity': 0,
                'morph_mean_solidity': 0,
                'morph_total_stained_area': 0,
                'morph_mean_compactness': 0
            }
        
        # Compute shape features for each biomarker region
        areas = [r.area for r in region_props]
        perimeters = [r.perimeter for r in region_props]
        compactness = [p**2 / (4*np.pi*a) if a > 0 else 0 
                      for p, a in zip(perimeters, areas)]
        
        features = {
            'morph_num_regions': len(region_props),
            'morph_mean_area': np.mean(areas),
            'morph_mean_perimeter': np.mean(perimeters),
            'morph_mean_eccentricity': np.mean([r.eccentricity for r in region_props]),
            'morph_mean_solidity': np.mean([r.solidity for r in region_props]),
            'morph_total_stained_area': sum(areas),
            'morph_mean_compactness': np.mean(compactness)
        }
        return features
    
    @staticmethod
    def compute_spatial_features(biomarker_mask):
        """Compute spatial distribution features of biomarker staining.
        
        Args:
            biomarker_mask: Boolean mask indicating biomarker presence
            
        Returns:
            dict: Dictionary of spatial features
        """
        y_coords, x_coords = np.nonzero(biomarker_mask)
        
        if len(x_coords) == 0:
            return {
                'spatial_mean_x': 0,
                'spatial_mean_y': 0,
                'spatial_std_x': 0,
                'spatial_std_y': 0,
                'spatial_spread': 0,
                'spatial_density': 0,
                'spatial_clustering': 0
            }
        
        # Compute basic spatial statistics
        features = {
            'spatial_mean_x': np.mean(x_coords),
            'spatial_mean_y': np.mean(y_coords),
            'spatial_std_x': np.std(x_coords),
            'spatial_std_y': np.std(y_coords),
            'spatial_spread': np.sqrt(np.var(x_coords) + np.var(y_coords)),
            'spatial_density': len(x_coords) / biomarker_mask.size
        }
        
        # Compute clustering using nearest neighbor distances
        if biomarker_mask is not None:
            # Ensure biomarker_mask is a boolean array
            bool_mask = biomarker_mask.astype(bool)
            if np.any(bool_mask):
                dist_transform = ndimage.distance_transform_edt(~bool_mask)
                if dist_transform is not None:
                    features['spatial_clustering'] = np.mean(dist_transform[bool_mask])
                else:
                    features['spatial_clustering'] = 0
            else:
                features['spatial_clustering'] = 0
        else:
            features['spatial_clustering'] = 0
        
        return features
    
    @staticmethod
    def compute_intensity_features(image_patch, biomarker_mask):
        """Compute intensity statistics of biomarker regions.
        
        Args:
            image_patch: 2D numpy array of image intensities
            biomarker_mask: Boolean mask indicating biomarker presence
            
        Returns:
            dict: Dictionary of intensity features
        """
        if not np.any(biomarker_mask):
            return {
                'intensity_mean': 0,
                'intensity_std': 0,
                'intensity_q25': 0,
                'intensity_median': 0,
                'intensity_q75': 0,
                'intensity_skewness': 0,
                'intensity_kurtosis': 0
            }
        
        biomarker_intensities = image_patch[biomarker_mask]
        
        features = {
            'intensity_mean': np.mean(biomarker_intensities),
            'intensity_std': np.std(biomarker_intensities),
            'intensity_q25': np.percentile(biomarker_intensities, 25),
            'intensity_median': np.percentile(biomarker_intensities, 50),
            'intensity_q75': np.percentile(biomarker_intensities, 75),
            'intensity_skewness': float(stats.skew(biomarker_intensities)),
            'intensity_kurtosis': float(stats.kurtosis(biomarker_intensities))
        }
        return features
    
    def extract_all_features(self, image_patch, biomarker_mask):
        """Extract all features from biomarker regions.
        
        Args:
            image_patch: 2D numpy array of image intensities
            biomarker_mask: Boolean mask indicating biomarker presence
            
        Returns:
            dict: Dictionary containing all computed features
        """
        if not np.any(biomarker_mask):
            return {}
        
        features = {}
        
        # Compute all feature types
        features.update(self.compute_texture_features(image_patch, biomarker_mask))
        features.update(self.compute_morphological_features(biomarker_mask))
        features.update(self.compute_spatial_features(biomarker_mask))
        features.update(self.compute_intensity_features(image_patch, biomarker_mask))
        
        return features