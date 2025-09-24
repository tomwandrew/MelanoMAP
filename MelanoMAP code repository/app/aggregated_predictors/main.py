import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

GT_COLUMNS = ['AMBLor Score', 'FinalFilename']
RESULTS_COLUMNS = ['File Name', 'HC Prediction', 'HC P-value']

def load_and_preprocess_data(bcn_path, bel_path, results_path):
    """
    Load and preprocess the data from CSV files.
    """
    # Load ground truth data
    bcn_gt = pd.read_csv(bcn_path)[GT_COLUMNS]
    bel_gt = pd.read_csv(bel_path)[GT_COLUMNS]
    
    # Convert AMBLor Score to binary
    bcn_gt['AMBLor Score'] = bcn_gt['AMBLor Score'].replace({
        'At Risk': 1,
        'Low Risk': 0
    })
    
    # Combine and process ground truth data
    gt = pd.concat([bcn_gt, bel_gt])
    gt['filename'] = gt['FinalFilename'].apply(lambda x: x.split('.')[-2])
    gt = gt.drop(columns=['FinalFilename'])
    
    # Load and process results data
    results = pd.read_csv(results_path)[RESULTS_COLUMNS]
    results['filename'] = results['File Name']
    results = results.drop(columns=['File Name'])
    
    # Merge datasets
    merged = pd.merge(gt, results, on='filename', how='inner')
    
    # Convert to numeric and handle NaN values
    merged['AMBLor Score'] = pd.to_numeric(merged['AMBLor Score'], errors='coerce')
    merged['HC P-value'] = pd.to_numeric(merged['HC P-value'], errors='coerce')
    merged = merged.dropna(subset=['AMBLor Score', 'HC P-value'])
    
    # Transform p-values
    merged['HC P-value transformed'] = -np.log10(merged['HC P-value'] + 1e-300)
    
    return merged

def plot_distributions(low_risk_data, high_risk_data, threshold=None, output_path='distributions.png'):
    """
    Plot overlapping distributions with optional threshold line.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    plt.hist(low_risk_data, bins=30, alpha=0.5, color='green', density=True, 
             label=f'Low Risk (n={len(low_risk_data)})')
    plt.hist(high_risk_data, bins=30, alpha=0.5, color='red', density=True, 
             label=f'At Risk (n={len(high_risk_data)})')
    
    # Add KDE curves
    sns.kdeplot(data=low_risk_data, color='darkgreen', linewidth=2)
    sns.kdeplot(data=high_risk_data, color='darkred', linewidth=2)
    
    # Add threshold line if provided
    if threshold is not None:
        plt.axvline(x=threshold, color='black', linestyle='--', 
                   label=f'Threshold = {threshold:.3f}')
    
    # Add labels and statistics
    plt.xlabel('-log10(HC P-value)')
    plt.ylabel('Density')
    plt.title('Distribution of -log10(HC P-values) by AMBLor Score')
    
    stats_text = (f'Low Risk: mean={low_risk_data.mean():.3f}, std={low_risk_data.std():.3f}\n'
                 f'At Risk: mean={high_risk_data.mean():.3f}, std={high_risk_data.std():.3f}')
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, output_path='roc_curve.png'):
    """
    Plot ROC curve with AUC score.
    """
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def find_optimal_threshold(merged_data):
    """
    Calculate ROC curve and find optimal threshold using Youden's J statistic.
    """
    fpr, tpr, thresholds = roc_curve(merged_data['AMBLor Score'], 
                                    merged_data['HC P-value transformed'])
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, fpr, tpr, roc_auc

def evaluate_model(merged_data, threshold):
    """
    Evaluate the model using the given threshold and print metrics.
    """
    predictions = (merged_data['HC P-value transformed'] >= threshold).astype(int)
    
    print("\nOptimal threshold:", threshold)
    print("\nConfusion Matrix:")
    print(confusion_matrix(merged_data['AMBLor Score'], predictions))
    print("\nClassification Report:")
    print(classification_report(merged_data['AMBLor Score'], predictions))
    
    return predictions

def main():
    # Load and preprocess data
    merged = load_and_preprocess_data(
        'data/final_bcn_score_filename.csv',
        'data/final_bel_score_filename.csv',
        'data/processed_slides_results_august_am.csv'
    )
    
    # Split data by risk groups
    low_risk_data = merged[merged['AMBLor Score'] == 0]['HC P-value transformed']
    high_risk_data = merged[merged['AMBLor Score'] == 1]['HC P-value transformed']
    
    # Find optimal threshold and calculate ROC curve
    optimal_threshold, fpr, tpr, roc_auc = find_optimal_threshold(merged)
    
    # Create plots
    plot_roc_curve(fpr, tpr, roc_auc)
    plot_distributions(low_risk_data, high_risk_data, optimal_threshold, 
                      'hc_pvalue_distributions_with_threshold.png')
    
    # Evaluate model and make predictions
    predictions = evaluate_model(merged, optimal_threshold)
    
    # Save results
    merged['Predicted_Class'] = predictions
    merged.to_csv('results_with_predictions.csv', index=False)
    print("\nResults saved to 'results_with_predictions.csv'")

if __name__ == '__main__':
    main()