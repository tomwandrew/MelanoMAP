"""Main module for AM marker analysis."""
from glob import glob
import pandas as pd
from .am_analyzer import AMAnalyzer

def hc_am_detection(image_path, clean_patches = None):
    """
    Analyze AM marker expression in histological images.
    
    Args:
        image_path: Path to directory containing image files
        clean_patches: DataFrame containing clean patch information
        
    Returns:
        tuple: (patch_stats DataFrame, prediction string, p-value float)
    """
    # Initialize analyzer
    analyzer = AMAnalyzer()
    
    # Process files
    files = pd.DataFrame(glob(f'{image_path}/**/*.jpg', recursive=True), 
                        columns=['img_path'])
    
    if len(files) == 0:
        print(f"No image files found in {image_path}")
        return pd.DataFrame(), "Cannot determine", 1.0
        
    files['image_name'] = files['img_path'].str.extract(r'([^\/]+)\.jpg$')[0]
    files['slide'] = files['img_path'].str.split('/').str[-2]
    

    if clean_patches is not None:
    # Filter for clean patches
        files = files[files['image_name'].isin(clean_patches['image_name'])]
        
    
    if len(files) == 0:
        print("No files match the clean patches criteria")
        return pd.DataFrame(), "Cannot determine", 1.0
        
    files['mask_path'] = 'results_pipeline/masks/' + files['slide'] + '/' + \
                        files['image_name'] + '.png'
    
    # Get slide name
    try:
        slide_name = files['slide'].iloc[0]
    except IndexError:
        print("No valid slides found")
        return pd.DataFrame(), "Cannot determine", 1.0
    
    print(f"Processing slide {slide_name} with {len(files)} patches")
    
    # Get ground truth
    try:
        # Load BCN and BEL data
        bcn_gt = pd.read_csv('GT_Comparison/data/final_bcn_score_filename.csv')[['AMBLor Score', 'FinalFilename']]
        bel_gt = pd.read_csv('GT_Comparison/data/final_bel_score_filename.csv')[['AMBLor Score', 'FinalFilename']]
        
        # Combine ground truth data
        gt = pd.concat([bcn_gt, bel_gt])
        gt['slide'] = gt['FinalFilename'].apply(lambda x: x.split('.')[-2])
        
        # Convert AMBLor Score to binary
        gt['AMBLor Score'] = gt['AMBLor Score'].replace({
            'At Risk': 1,
            'Low Risk': 0
        })
        
        # Get ground truth for this slide
        slide_gt_data = gt[gt['slide'] == slide_name]
        if len(slide_gt_data) > 0:
            slide_gt = slide_gt_data.iloc[0]
            print(f"Found ground truth for slide {slide_name}: {slide_gt['AMBLor Score']}")
        else:
            print(f"No ground truth found for slide {slide_name}")
            slide_gt = None
    except Exception as e:
        print(f"Error loading ground truth data: {str(e)}")
        slide_gt = None
    
    # Analyze patches
    patch_results = []
    for _, row in files.iterrows():
        stats = analyzer.analyze_patch(row['img_path'], row['mask_path'])
        if stats:
            stats['image_name'] = row['image_name']
            patch_results.append(stats)
    
    if not patch_results:
        print(f"No valid patches found for slide {slide_name}")
        return pd.DataFrame(), "Cannot determine", 1.0
    
    # Create patch-level DataFrame
    patch_stats = pd.DataFrame(patch_results)
    
    # Analyze slide
    analysis_results = analyzer.analyze_slide(patch_stats)
    if not analysis_results:
        print(f"Analysis failed for slide {slide_name}")
        return patch_stats, "Cannot determine", 1.0
    
    # Plot results with ground truth
    analyzer.plot_results(patch_stats, analysis_results, slide_name, slide_gt)
    
    # Save statistics
    slide_dir = analyzer._create_slide_directory(slide_name)
    patch_stats.to_csv(f'{slide_dir}/patch_statistics.csv', index=False)
    
    # Get results
    prediction = ('Lost expression' if analysis_results['summary']['lost_expression'] 
                 else 'Maintained expression')
    p_value = float(analysis_results['summary']['p_value'])
    
    # Print results with ground truth comparison
    print(f"\nSlide {slide_name} Analysis Results:")
    print(f"Prediction: {prediction}")
    print(f"P-value: {p_value:.4f}")
    print(f"Effect size: {analysis_results['summary']['effect_size']:.4f}")
    print(f"Relative change: {analysis_results['summary']['relative_change']*100:.1f}%")
    
    if slide_gt is not None:
        gt_risk = "At Risk" if slide_gt['AMBLor Score'] == 1 else "Low Risk"
        print(f"Ground Truth: {gt_risk}")
        print(f"Match: {(gt_risk == 'At Risk') == analysis_results['summary']['lost_expression']}")
    
    return patch_stats, prediction, p_value

if __name__ == '__main__':
    hc_am_detection('test_path', pd.DataFrame())