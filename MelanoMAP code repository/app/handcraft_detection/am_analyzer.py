"""AM marker analysis implementation."""
from .base_analyzer import BaseAnalyzer
from .statistical_analyzer import StatisticalAnalyzer
from .visualization import Visualizer
from .feature_extractor import FeatureExtractor
from skimage.color import rgb2hsv
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(filename='am_analysis_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

REQUIRED_FEATURES = [
    'am_mean', 'intensity_kurtosis', 'intensity_std', 'texture_dissimilarity', 
    'morph_mean_area', 'morph_mean_perimeter', 'spatial_density'
]


class AMAnalyzer(BaseAnalyzer):
    def __init__(self, output_dir='results_pipeline'):
        super().__init__(output_dir)
        self.base_dir = f'{output_dir}/handcraft/am'
        self.feature_extractor = FeatureExtractor()
        self._create_directories(self.base_dir)
    
    def _create_slide_directory(self, slide_name):
        """Create and return slide-specific directory structure."""
        slide_dir = f'{self.base_dir}/{slide_name}'
        self._create_directories(
            slide_dir,
            f'{slide_dir}/visualizations',
            f'{slide_dir}/feature_analysis'
        )
        return slide_dir
    
    def analyze_patch(self, img_path, mask_path):
        """Analyze AM expression in a single patch, focusing only on AM-positive regions."""
        try:
            img, mask = self.load_image_and_mask(img_path, mask_path)
            value_channel = rgb2hsv(img)[:, :, 2]
            
            # Create masks
            am_mask = mask == 3  # AM staining regions
            tumor_mask = mask == 2  # Tumor regions
            
            # Check if we have any AM staining
            if not np.any(am_mask):
                return None
            
            # Basic AM intensity in AM regions
            features = {
                'has_tumor': np.any(tumor_mask),  # Only used for grouping
                'am_pixels': np.sum(am_mask)  # Total AM-positive pixels
            }
            
            # Extract all features from AM regions only
            am_features = self.feature_extractor.extract_all_features(
                value_channel,  # Intensity image
                am_mask  # Only analyze AM-positive regions
            )
            
            # Add extracted features
            if am_features:  # Only add if we got valid features
                features.update(am_features)
                # Use intensity_mean as am_mean for consistency
                features['am_mean'] = features.get('intensity_mean', 0)
            
            return features
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None
    
    def analyze_slide(self, patch_data):
        """
        Analyze all features in AM regions, comparing tumor vs non-tumor areas.
        
        Args:
            patch_data: DataFrame containing patch-level features
        
        Returns:
            dict: Analysis results including statistics for all features
        """
        if len(patch_data) == 0:
            return None
            
        # Get feature columns (excluding non-feature columns)
        non_feature_cols = ['image_name', 'has_tumor']
        feature_cols = [col for col in patch_data.columns if col not in non_feature_cols]
        
        analysis_results = {
            'features': {},
            'summary': {}
        }
        
        # Analyze each feature
        p_values = []
        feature_results_list = []
        
        for feature in feature_cols:
            tumor_values = patch_data[patch_data['has_tumor']][feature].dropna()
            non_tumor_values = patch_data[~patch_data['has_tumor']][feature].dropna()
            
            if len(tumor_values) == 0 or len(non_tumor_values) == 0:
                continue
                
            try:
                # Statistical test
                u_stat, p_value = stats.mannwhitneyu(
                    tumor_values,
                    non_tumor_values,
                    alternative='two-sided'
                )
                
                # Effect size calculation with safeguards
                tumor_mean = float(np.mean(tumor_values))
                non_tumor_mean = float(np.mean(non_tumor_values))
                tumor_std = float(np.std(tumor_values))
                non_tumor_std = float(np.std(non_tumor_values))
                
                # Check for valid standard deviations
                pooled_std = np.sqrt((tumor_std**2 + non_tumor_std**2) / 2)
                if pooled_std == 0:
                    effect_size = 0.0
                else:
                    effect_size = (tumor_mean - non_tumor_mean) / pooled_std
                
                # Calculate relative change with safeguard
                if non_tumor_mean == 0:
                    relative_change = 0.0
                else:
                    relative_change = (tumor_mean - non_tumor_mean) / abs(non_tumor_mean)
                
                feature_results = {
                    'p_value': float(p_value),
                    'effect_size': float(effect_size),
                    'tumor_mean': tumor_mean,
                    'non_tumor_mean': non_tumor_mean,
                    'relative_change': float(relative_change),
                    'tumor_count': len(tumor_values),
                    'non_tumor_count': len(non_tumor_values),
                    'significant': p_value < 0.05
                }
                
                analysis_results['features'][feature] = feature_results
                
                # Track significant features
                if p_value < 0.05 and abs(effect_size) > 0.5:
                    significant_features.append({
                        'feature': feature,
                        'effect_size': effect_size,
                        'p_value': p_value,
                        'relative_change': relative_change
                    })
                
            except Exception as e:
                print(f"Analysis failed for feature {feature}: {str(e)}")
                continue
        
        # Sort significant features by absolute effect size
        significant_features.sort(key=lambda x: abs(x['effect_size']), reverse=True)
        
        # Overall analysis summary
        am_mean_results = analysis_results['features'].get('am_mean', {})
        kurtosis_results = analysis_results['features'].get('intensity_kurtosis', {})
        
        # Determine if expression is lost with safeguards
        lost_expression = (
            am_mean_results.get('p_value', 1.0) < 0.05 and
            am_mean_results.get('effect_size', 0) > 0 and
            am_mean_results.get('relative_change', 0) > 0.03
        )
        
        analysis_results['summary'] = {
            'p_value': float(am_mean_results.get('p_value', 1.0)),
            'effect_size': float(am_mean_results.get('effect_size', 0)),
            'tumor_mean': float(am_mean_results.get('tumor_mean', 0)),
            'non_tumor_mean': float(am_mean_results.get('non_tumor_mean', 0)),
            'relative_change': float(am_mean_results.get('relative_change', 0)),
            'kurtosis_p_value': float(kurtosis_results.get('p_value', 1.0)),
            'kurtosis_effect_size': float(kurtosis_results.get('effect_size', 0)),
            'lost_expression': bool(lost_expression),
            'significant_features': significant_features
        }
        
        return analysis_results
    
    def plot_results(self, patch_stats, analysis_results, slide_name, ground_truth=None):
        """Plot and save comprehensive analysis results."""
        if len(patch_stats) == 0:
            print(f"No data to plot for slide {slide_name}")
            return
        
        slide_dir = self._create_slide_directory(slide_name)
        vis_dir = f'{slide_dir}/visualizations'
        
        # Plot main AM analysis
        Visualizer.plot_am_analysis(
            patch_stats, 
            analysis_results, 
            f'{vis_dir}/am_expression_analysis.png',
            ground_truth
        )
        
        # Plot all features analysis
        Visualizer.plot_feature_analysis(
            patch_stats,
            analysis_results,
            f'{vis_dir}/feature_analysis.png'
        )
        
        # Plot detailed feature comparison
        Visualizer.plot_feature_comparison(
            patch_stats,
            analysis_results,
            f'{vis_dir}/feature_comparison.png'
        )
        
        # Save detailed statistics
        patch_stats.to_csv(f'{slide_dir}/patch_statistics.csv', index=False)
        
        # Save feature analysis results
        feature_analysis = pd.DataFrame([
            {
                'feature': feature,
                **analysis_results['features'][feature]
            }
            for feature in analysis_results['features']
        ])
        
        # Sort by absolute effect size to see most discriminative features
        feature_analysis['abs_effect_size'] = abs(feature_analysis['effect_size'])
        feature_analysis = feature_analysis.sort_values('abs_effect_size', ascending=False)
        
        feature_analysis.to_csv(f'{slide_dir}/feature_analysis.csv', index=False)
        
        # Print top features
        print("\nTop discriminative features:")
        for _, row in feature_analysis.head().iterrows():
            print(f"\n{row['feature']}:")
            print(f"  Effect size: {row['effect_size']:.3f}")
            print(f"  P-value: {row['p_value']:.2e}")
            print(f"  Relative change: {row['relative_change']*100:.1f}%")