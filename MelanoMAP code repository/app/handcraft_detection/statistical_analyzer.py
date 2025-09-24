"""Statistical analysis utilities for histological images."""
from scipy import stats
import numpy as np
import pandas as pd

class StatisticalAnalyzer:
    @staticmethod
    def compute_basic_stats(values1, values2):
        """Compute basic statistical comparison between two groups."""
        if len(values1) == 0 or len(values2) == 0:
            return None
            
        try:
            # Statistical test
            u_stat, p_value = stats.mannwhitneyu(
                values1, values2, alternative='two-sided'
            )
            
            # Effect size
            mean1, mean2 = np.mean(values1), np.mean(values2)
            pooled_std = np.sqrt((np.std(values1)**2 + np.std(values2)**2) / 2)
            effect_size = (mean1 - mean2) / pooled_std if pooled_std != 0 else 0
            
            # Relative change
            relative_change = (mean1 - mean2) / abs(mean2) if mean2 != 0 else 0
            
            return {
                'p_value': float(p_value),
                'effect_size': float(effect_size),
                'mean1': float(mean1),
                'mean2': float(mean2),
                'relative_change': float(relative_change),
                'n1': len(values1),
                'n2': len(values2)
            }
        except Exception as e:
            print(f"Statistical analysis failed: {str(e)}")
            return None
    
    @staticmethod
    def analyze_features(data, group_col, feature_cols):
        """Analyze differences in features between groups."""
        results = []
        
        for feature in feature_cols:
            group1 = data[data[group_col]][feature]
            group2 = data[~data[group_col]][feature]
            
            stats = StatisticalAnalyzer.compute_basic_stats(group1, group2)
            if stats:
                stats['feature'] = feature
                results.append(stats)
        
        return pd.DataFrame(results) 