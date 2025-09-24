import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def load_ground_truth():
    """Load and combine ground truth data."""
    # Load ground truth data with clinical features including Met
    # bcn_gt = pd.read_csv('data/final_bcn_score_filename.csv')[
    #     ['AMBLor Score', 'FinalFilename', 'Breslow', 'Age', 'Stage', 'Gender', 'DFS', 'Met']
    # ]
    bel_gt = pd.read_csv('data/final_bel_score_filename.csv')[
        ['AMBLor Score', 'FinalFilename', 'Breslow', 'Age', 'Stage', 'Gender', 'DFS', 'Met']
    ]
    
    # Combine ground truth data
    # gt = pd.concat([bcn_gt, bel_gt])
    gt = bel_gt
    gt['slide'] = gt['FinalFilename'].apply(lambda x: x.split('.')[-2])
    
    # Convert AMBLor Score to binary (numeric)
    gt['AMBLor Score'] = (gt['AMBLor Score'] == 'At Risk').astype(int)
    
    return gt

def load_feature_analyses(base_dir='../results_pipeline/handcraft/am'):
    """Load feature analysis results for all slides."""
    feature_data = []
    
    # Find all feature analysis files
    feature_files = glob(f'{base_dir}/**/feature_analysis.csv', recursive=True)
    
    if not feature_files:
        print(f"No feature analysis files found in {base_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(feature_files)} feature analysis files")
    
    for feature_file in feature_files:
        try:
            # Check if file exists and is not empty
            if os.path.getsize(feature_file) == 0:
                print(f"Skipping empty file: {feature_file}")
                continue
                
            slide_name = feature_file.split('/')[-2]  # Get slide name from path
            print(f"Processing {slide_name} from {feature_file}")
            
            # Try reading the CSV file with error handling
            try:
                slide_features = pd.read_csv(feature_file)
            except pd.errors.EmptyDataError:
                print(f"Empty CSV file: {feature_file}")
                continue
            except Exception as e:
                print(f"Error reading {feature_file}: {str(e)}")
                continue
            
            # Check if DataFrame is empty or missing required columns
            if slide_features.empty:
                print(f"Empty DataFrame for {slide_name}")
                continue
                
            required_columns = ['feature', 'effect_size', 'p_value']
            if not all(col in slide_features.columns for col in required_columns):
                print(f"Missing required columns in {slide_name}")
                continue
            
            # Add slide information and AIPrediction (default to None since AM doesn't have predictions)
            slide_features['slide'] = slide_name
            slide_features['AIPrediction'] = None
            feature_data.append(slide_features)
            print(f"Successfully loaded data for {slide_name}")
            
        except Exception as e:
            print(f"Error processing {feature_file}: {str(e)}")
            continue
    
    if not feature_data:
        print("No valid feature data found")
        return pd.DataFrame()
    
    # Combine all feature data
    combined_data = pd.concat(feature_data, ignore_index=True)
    print(f"Total features loaded: {len(combined_data)}")
    
    return combined_data

def analyze_feature_performance(feature_data, ground_truth):
    """Analyze how well each feature predicts the ground truth."""
    # Merge feature data with ground truth including Met
    merged_data = pd.merge(feature_data, 
                          ground_truth[['slide', 'AMBLor Score', 'Breslow', 'Age', 
                                      'Stage', 'Gender', 'DFS', 'Met']], 
                          on='slide', how='inner')
    
    # Rename AMBLor Score to PathologyScore
    merged_data = merged_data.rename(columns={'AMBLor Score': 'PathologyScore'})
    
    if merged_data.empty:
        print("No matching data found between features and ground truth!")
        print(f"Feature data slides: {feature_data['slide'].nunique()}")
        print(f"Ground truth slides: {ground_truth['slide'].nunique()}")
        print(f"Sample feature slides: {feature_data['slide'].head().tolist()}")
        print(f"Sample ground truth slides: {ground_truth['slide'].head().tolist()}")
        return pd.DataFrame()
    
    # Save all data formats
    output_dir = 'results_pipeline/feature_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save the raw merged data (long format)
    merged_data.to_csv(f'{output_dir}/feature_data_raw_am.csv', index=False)
    
    # 2. Save wide format with all features and clinical data including Met
    wide_data = merged_data.pivot(index=['slide', 'PathologyScore', 'AIPrediction', 
                                       'Breslow', 'Age', 'Stage', 'Gender', 'DFS', 'Met'], 
                                columns='feature', 
                                values='effect_size').reset_index()
    
    # Ensure PathologyScore is binary
    wide_data['PathologyScore'] = wide_data['PathologyScore'].astype(int)
    wide_data.to_csv(f'{output_dir}/feature_data_wide_am.csv', index=False)
    
    # 3. Save feature-wise statistics
    feature_stats = merged_data.groupby('feature').agg({
        'effect_size': ['mean', 'std', 'min', 'max', 'count'],
        'p_value': ['mean', 'min', 'max']
    }).reset_index()
    feature_stats.columns = ['feature', 'mean_effect', 'std_effect', 'min_effect', 'max_effect', 
                           'count', 'mean_pvalue', 'min_pvalue', 'max_pvalue']
    feature_stats.to_csv(f'{output_dir}/feature_statistics_am.csv', index=False)
    
    # 4. Save correlation between features and ground truth
    correlations = []
    for feature in merged_data['feature'].unique():
        feature_data = merged_data[merged_data['feature'] == feature]
        correlation = feature_data['effect_size'].corr(feature_data['PathologyScore'])
        correlations.append({
            'feature': feature,
            'correlation_with_ground_truth': correlation
        })
    correlation_df = pd.DataFrame(correlations)
    correlation_df.to_csv(f'{output_dir}/feature_correlations_am.csv', index=False)
    
    # Continue with the performance analysis...
    feature_performance = []
    
    for feature in merged_data['feature'].unique():
        feature_subset = merged_data[merged_data['feature'] == feature]
        
        # Skip features with insufficient data
        if len(feature_subset) < 10:
            continue
        
        try:
            # Calculate ROC curve and AUC
            fpr, tpr, thresholds = roc_curve(feature_subset['PathologyScore'], 
                                           feature_subset['effect_size'])
            roc_auc = auc(fpr, tpr)
            
            # If AUC < 0.5, invert the predictions and recalculate
            if roc_auc < 0.5:
                fpr, tpr, thresholds = roc_curve(feature_subset['PathologyScore'], 
                                               -feature_subset['effect_size'])
                roc_auc = auc(fpr, tpr)
                inverted = True
            else:
                inverted = False
            
            # Find optimal threshold using Youden's J statistic
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            # Make predictions using optimal threshold
            if inverted:
                predictions = (-feature_subset['effect_size'] > optimal_threshold).astype(int)
            else:
                predictions = (feature_subset['effect_size'] > optimal_threshold).astype(int)
            
            # Calculate accuracy
            accuracy = (predictions == feature_subset['PathologyScore']).mean()
            
            # Get confusion matrix
            tn, fp, fn, tp = confusion_matrix(feature_subset['PathologyScore'], 
                                            predictions).ravel()
            
            # Calculate sensitivity and specificity
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            feature_performance.append({
                'feature': feature,
                'auc': roc_auc,
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'optimal_threshold': optimal_threshold,
                'inverted': inverted,
                'n_samples': len(feature_subset)
            })
            
        except Exception as e:
            print(f"Error analyzing feature {feature}: {str(e)}")
            continue
    
    return pd.DataFrame(feature_performance)

def plot_feature_performance(performance_df, output_dir='results_pipeline/feature_analysis'):
    """Create visualizations of feature performance."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort features by AUC
    performance_df = performance_df.sort_values('auc', ascending=False)
    
    # Plot AUC scores
    plt.figure(figsize=(12, 6))
    bars = sns.barplot(data=performance_df, x='feature', y='auc')
    
    # Add markers for inverted features
    for i, row in performance_df.iterrows():
        if row['inverted']:
            bars.text(i, row['auc'], '*', ha='center', va='bottom', color='red')
    
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Performance (AUC)\n* indicates inverted relationship')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random classifier')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_auc_scores.png')
    plt.close()
    
    # Print results with direction information
    print("\nFeature Performance Details:")
    for _, row in performance_df.iterrows():
        direction = "inversely " if row['inverted'] else "directly "
        print(f"\nFeature: {row['feature']}")
        print(f"AUC: {row['auc']:.3f} ({direction}predictive)")
        print(f"Accuracy: {row['accuracy']:.3f}")
        print(f"Sensitivity: {row['sensitivity']:.3f}")
        print(f"Specificity: {row['specificity']:.3f}")


def main():
    # Load data
    print("Loading ground truth data...")
    ground_truth = load_ground_truth()
    
    if ground_truth.empty:
        print("Error: No ground truth data loaded")
        return
    
    print(f"Loaded ground truth data for {len(ground_truth)} slides")
    
    print("\nLoading feature analyses...")
    feature_data = load_feature_analyses()
    
    if feature_data.empty:
        print("Error: No feature data loaded")
        return
    
    print(f"Loaded feature data with {len(feature_data)} entries")
    
    print("\nAnalyzing feature performance...")
    performance = analyze_feature_performance(feature_data, ground_truth)
    
    if performance.empty:
        print("Error: No performance data generated")
        return
    
    # Sort by AUC and print results
    performance = performance.sort_values('auc', ascending=False)
    print("\nFeature Performance (sorted by AUC):")
    print(performance)
    
    # Save results
    output_dir = 'results_pipeline/feature_analysis'
    os.makedirs(output_dir, exist_ok=True)
    performance.to_csv(f'{output_dir}/feature_performance.csv', index=False)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_feature_performance(performance, output_dir)
    
    # Print top 5 features
    print("\nTop 5 performing features:")
    top_features = performance.head()
    for _, row in top_features.iterrows():
        print(f"\nFeature: {row['feature']}")
        print(f"AUC: {row['auc']:.3f}")
        print(f"Accuracy: {row['accuracy']:.3f}")
        print(f"Sensitivity: {row['sensitivity']:.3f}")
        print(f"Specificity: {row['specificity']:.3f}")
        print(f"Optimal threshold: {row['optimal_threshold']:.3f}")
        print(f"Number of samples: {row['n_samples']}")
    

if __name__ == "__main__":
    main()
