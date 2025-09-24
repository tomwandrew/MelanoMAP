"""Visualization utilities for histological analysis."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import pandas as pd

class Visualizer:
    @staticmethod
    def create_overlay_image(img, mask, colors):
        """Create an overlay visualization of regions."""
        overlay = img.copy()
        for value, color in colors.items():
            overlay[mask == value] = color
        return cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    
    @staticmethod
    def plot_feature_distributions(data, group_col, feature_col, ax, title=None):
        """Plot distributions of a feature by group."""
        sns.boxplot(data=data, x=group_col, y=feature_col, ax=ax)
        if title:
            ax.set_title(title)
    
    @staticmethod
    def plot_correlation_matrix(data, feature_cols, output_path):
        """Plot correlation matrix of features."""
        plt.figure(figsize=(15, 15))
        
        # Compute correlation matrix
        correlation_matrix = data[feature_cols].corr()
        
        # Replace NaN values with 0 to avoid formatting warnings
        correlation_matrix = correlation_matrix.fillna(0)
        
        # Create heatmap with custom format function
        sns.heatmap(correlation_matrix, 
                    annot=True, 
                    cmap='coolwarm', 
                    center=0,
                    fmt='.2f',  # Fixed format for all values
                    mask=np.zeros_like(correlation_matrix, dtype=bool))  # No mask
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_feature_analysis(patch_stats, analysis_results, output_path):
        """Plot analysis for each feature."""
        # Get all features except non-feature columns
        feature_cols = [col for col in patch_stats.columns 
                       if col not in ['image_name', 'has_tumor']]
        
        # Calculate number of rows needed (3 features per row)
        n_rows = (len(feature_cols) + 2) // 3
        
        # Create figure
        plt.figure(figsize=(20, 7*n_rows))
        
        # Plot each feature
        for idx, feature in enumerate(feature_cols):
            # Create subplot
            plt.subplot(n_rows, 3, idx + 1)
            
            # Boxplot
            sns.boxplot(data=patch_stats, x='has_tumor', y=feature)
            plt.title(f'{feature} Distribution')
            plt.xticks([0, 1], ['Non-tumor', 'Tumor'])
            
            # Add statistics if available
            if feature in analysis_results['features']:
                stats = analysis_results['features'][feature]
                stats_text = (
                    f"p-value: {stats['p_value']:.2e}\n"
                    f"effect size: {stats['effect_size']:.2f}\n"
                    f"relative change: {stats['relative_change']*100:.1f}%"
                )
                plt.text(0.05, 0.95, stats_text,
                        transform=plt.gca().transAxes,
                        verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    @staticmethod
    def plot_am_analysis(patch_stats, analysis_results, output_path, ground_truth=None):
        """Plot main AM expression analysis with ground truth comparison."""
        expression_status = "Lost Expression" if analysis_results['summary']['lost_expression'] else "Maintained Expression"
        status_color = 'red' if analysis_results['summary']['lost_expression'] else 'green'
        
        # Add ground truth information if available
        if ground_truth is not None:
            gt_status = "At Risk" if ground_truth['AMBLor Score'] == 1 else "Low Risk"
            title_suffix = f"\n(Ground Truth: {gt_status})"
            match_status = "Matches" if (
                (gt_status == "At Risk" and expression_status == "Lost Expression") or
                (gt_status == "Low Risk" and expression_status == "Maintained Expression")
            ) else "Does Not Match"
        else:
            title_suffix = ""
            match_status = ""
        
        # Create main figure for AM mean
        plt.figure(figsize=(15, 6))
        
        # AM Mean Boxplot
        plt.subplot(1, 2, 1)
        sns.boxplot(data=patch_stats, x='has_tumor', y='am_mean')
        plt.title(f'AM Mean Intensity Distribution\n({expression_status}){title_suffix}', 
                 color=status_color)
        plt.xticks([0, 1], ['Non-tumor', 'Tumor'])
        
        # Add statistics
        stats_text = (
            f"p-value: {analysis_results['summary']['p_value']:.2e}\n"
            f"effect size: {analysis_results['summary']['effect_size']:.2f}\n"
            f"relative change: {analysis_results['summary']['relative_change']*100:.1f}%\n"
            f"GT Match: {match_status}" if ground_truth is not None else
            f"p-value: {analysis_results['summary']['p_value']:.2e}\n"
            f"effect size: {analysis_results['summary']['effect_size']:.2f}\n"
            f"relative change: {analysis_results['summary']['relative_change']*100:.1f}%"
        )
        plt.text(0.05, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
        
        # AM Mean Histogram
        plt.subplot(1, 2, 2)
        tumor_data = patch_stats[patch_stats['has_tumor']]['am_mean']
        non_tumor_data = patch_stats[~patch_stats['has_tumor']]['am_mean']
        
        plt.hist(non_tumor_data, bins=30, alpha=0.5, color='blue', 
                label=f'Non-tumor (n={len(non_tumor_data)})')
        plt.hist(tumor_data, bins=30, alpha=0.5, color='red', 
                label=f'Tumor (n={len(tumor_data)})')
        
        plt.title('AM Mean Intensity Distribution')
        plt.xlabel('AM Mean Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    @staticmethod
    def plot_significant_features(patch_stats, significant_features, output_path):
        """Plot top significant features."""
        if not significant_features:
            return
            
        n_features = min(len(significant_features), 5)  # Top 5 most significant
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 4*n_features))
        if n_features == 1:
            axes = [axes]
        
        for i, feature_info in enumerate(significant_features[:n_features]):
            feature = feature_info['feature']
            Visualizer.plot_feature_distributions(
                patch_stats, 'has_tumor', feature, axes[i],
                f"{feature}\np={feature_info['p_value']:.2e}, effect={feature_info['effect_size']:.2f}"
            )
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    @staticmethod
    def save_analysis_report(slide_name, analysis_results, patch_stats, output_path):
        """Save a comprehensive text report of the analysis results."""
        tumor_data = patch_stats[patch_stats['has_tumor']]['am_mean']
        non_tumor_data = patch_stats[~patch_stats['has_tumor']]['am_mean']
        expression_status = "Lost Expression" if analysis_results['summary']['lost_expression'] else "Maintained Expression"
        
        with open(output_path, 'w') as f:
            f.write(f"Analysis Report for Slide {slide_name}\n")
            f.write("="*50 + "\n\n")
            
            # Overall Results
            f.write("Overall Results:\n")
            f.write("-"*20 + "\n")
            f.write(f"Expression Status: {expression_status}\n")
            f.write(f"P-value: {analysis_results['summary']['p_value']:.4e}\n")
            f.write(f"Effect Size: {analysis_results['summary']['effect_size']:.4f}\n")
            f.write(f"Relative Change: {analysis_results['summary']['relative_change']*100:.1f}%\n\n")
            
            # Patch Statistics
            f.write("Patch Statistics:\n")
            f.write("-"*20 + "\n")
            f.write(f"Total Patches: {len(patch_stats)}\n")
            f.write(f"Tumor Patches: {len(tumor_data)}\n")
            f.write(f"Non-tumor Patches: {len(non_tumor_data)}\n\n")
            
            # Feature Analysis
            f.write("Feature Analysis:\n")
            f.write("-"*20 + "\n")
            
            # Sort features by p-value
            feature_results = [
                {
                    'feature': feature,
                    **analysis_results['features'][feature]
                }
                for feature in analysis_results['features']
            ]
            feature_results.sort(key=lambda x: x['p_value'])
            
            # Significant Features
            f.write("\nSignificant Features (p < 0.05):\n")
            significant_features = [f for f in feature_results if f['p_value'] < 0.05]
            if significant_features:
                for feat in significant_features:
                    f.write(f"\n{feat['feature']}:\n")
                    f.write(f"  p-value: {feat['p_value']:.4e}\n")
                    f.write(f"  effect size: {feat['effect_size']:.4f}\n")
                    f.write(f"  relative change: {feat['relative_change']*100:.1f}%\n")
                    f.write(f"  tumor mean: {feat['tumor_mean']:.4f}\n")
                    f.write(f"  non-tumor mean: {feat['non_tumor_mean']:.4f}\n")
                    f.write(f"  samples: tumor={feat['tumor_count']}, non-tumor={feat['non_tumor_count']}\n")
            else:
                f.write("  None\n")
            
            # Non-significant Features
            f.write("\nNon-significant Features (p >= 0.05):\n")
            non_significant_features = [f for f in feature_results if f['p_value'] >= 0.05]
            if non_significant_features:
                for feat in non_significant_features:
                    f.write(f"\n{feat['feature']}:\n")
                    f.write(f"  p-value: {feat['p_value']:.4e}\n")
                    f.write(f"  effect size: {feat['effect_size']:.4f}\n")
                    f.write(f"  relative change: {feat['relative_change']*100:.1f}%\n")
                    f.write(f"  tumor mean: {feat['tumor_mean']:.4f}\n")
                    f.write(f"  non-tumor mean: {feat['non_tumor_mean']:.4f}\n")
                    f.write(f"  samples: tumor={feat['tumor_count']}, non-tumor={feat['non_tumor_count']}\n")
            else:
                f.write("  None\n")
            
            # Feature Categories Summary
            f.write("\nFeature Categories Summary:\n")
            f.write("-"*20 + "\n")
            categories = {
                'Intensity': ['intensity_', 'am_mean'],
                'Texture': ['texture_'],
                'Morphological': ['morph_'],
                'Spatial': ['spatial_']
            }
            
            for category, prefixes in categories.items():
                category_features = [
                    f for f in feature_results 
                    if any(f['feature'].startswith(p) for p in prefixes)
                ]
                significant_count = sum(1 for f in category_features if f['p_value'] < 0.05)
                f.write(f"\n{category} Features:\n")
                f.write(f"  Total: {len(category_features)}\n")
                f.write(f"  Significant: {significant_count}\n")
    
    @staticmethod
    def plot_detailed_feature(patch_stats, analysis_results, feature_name, output_path):
        """Create detailed plot for a specific feature."""
        plt.figure(figsize=(15, 6))
        
        # Boxplot
        plt.subplot(1, 2, 1)
        sns.boxplot(data=patch_stats, x='has_tumor', y=feature_name)
        plt.title(f'{feature_name} Distribution')
        plt.xticks([0, 1], ['Non-tumor', 'Tumor'])
        
        # Add statistics if available
        if feature_name in analysis_results['features']:
            stats = analysis_results['features'][feature_name]
            stats_text = (
                f"p-value: {stats['p_value']:.2e}\n"
                f"effect size: {stats['effect_size']:.2f}\n"
                f"relative change: {stats['relative_change']*100:.1f}%\n"
                f"tumor mean: {stats['tumor_mean']:.3f}\n"
                f"non-tumor mean: {stats['non_tumor_mean']:.3f}"
            )
            plt.text(0.05, 0.95, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # Histogram
        plt.subplot(1, 2, 2)
        tumor_data = patch_stats[patch_stats['has_tumor']][feature_name]
        non_tumor_data = patch_stats[~patch_stats['has_tumor']][feature_name]
        
        plt.hist(non_tumor_data, bins=30, alpha=0.5, color='blue', 
                label=f'Non-tumor (n={len(non_tumor_data)})')
        plt.hist(tumor_data, bins=30, alpha=0.5, color='red', 
                label=f'Tumor (n={len(tumor_data)})')
        
        plt.title(f'{feature_name} Distribution')
        plt.xlabel(feature_name)
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    @staticmethod
    def plot_feature_comparison(patch_stats, analysis_results, output_path):
        """Plot comparison of intensity_kurtosis and texture_dissimilarity."""
        features = ['intensity_kurtosis', 'texture_dissimilarity']
        
        for feature in features:
            Visualizer.plot_detailed_feature(
                patch_stats,
                analysis_results,
                feature,
                output_path.replace('.png', f'_{feature}.png')
            )