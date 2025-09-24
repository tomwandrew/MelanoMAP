import os
import numpy as np
import torch
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from PIL import Image
from tqdm import tqdm

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import os
import torch
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import label_binarize
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, valid_files, transform=None):
        self.image_paths = [
            os.path.join(image_dir, img) for img in os.listdir(image_dir)
            if any(img in path for path in valid_files)  # Filtering based on valid files
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        if self.transform:
            img = self.transform(image=img)['image']

        return img, img_path

def get_inference_augmentation():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def load_model(staining):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = models.efficientnet_v2_s()
    num_ftrs = model.classifier[1].in_features
    
    model.classifier[1] = torch.nn.Linear(num_ftrs, 2)

    model_path = f'models/best_classifier_{staining}.pth'
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
     
    model.to(device)
    model.eval()
    
    return model

def compute_prediction_statistics(prediction_scores_list):
    lost_expression_probs = [score['lost_expression_prob'] for score in prediction_scores_list]
    maintained_expression_probs = [score['maintained_expression_prob'] for score in prediction_scores_list]
    
    predictions = ['Lost expression' if score['lost_expression_prob'] > score['maintained_expression_prob'] 
                  else 'Maintained expression' for score in prediction_scores_list]
    
    stats = {
        'max_lost_expression_prob': max(lost_expression_probs),
        'min_lost_expression_prob': min(lost_expression_probs),
        'mean_lost_expression_prob': np.mean(lost_expression_probs),
        'std_lost_expression_prob': np.std(lost_expression_probs),
        'max_maintained_expression_prob': max(maintained_expression_probs),
        'min_maintained_expression_prob': min(maintained_expression_probs),
        'mean_maintained_expression_prob': np.mean(maintained_expression_probs),
        'std_maintained_expression_prob': np.std(maintained_expression_probs),
        'percentage_lost_expression': (sum(1 for p in predictions if p == 'Lost expression') / len(predictions)) * 100,
        'percentage_maintained_expression': (sum(1 for p in predictions if p == 'Maintained expression') / len(predictions)) * 100,
        'total_patches': len(prediction_scores_list)
    }
    
    return stats

def save_prediction_scores(prediction_scores_list, staining, image_dir, filename=None):
    # Create a DataFrame for the prediction scores
    prediction_scores_df = pd.DataFrame(prediction_scores_list)
    
    # Calculate statistics
    stats = compute_prediction_statistics(prediction_scores_list)
    stats_df = pd.DataFrame([stats])
    
    # Ensure the directory exists
    output_dir = f'/app/results_pipeline/ai_predictions/{staining}'
    os.makedirs(output_dir, exist_ok=True)

    # If we have not found any patches with LO
    if filename is None:
        filename = image_dir.split('/')[-1]

    base_filename = filename.split(" ")[0]
    
    # Save the prediction scores DataFrame to a CSV file
    scores_output_path = os.path.join(output_dir, f'{base_filename}_predictions.csv')
    prediction_scores_df.to_csv(scores_output_path, index=False)
    
    # Save the statistics DataFrame to a separate CSV file
    stats_output_path = os.path.join(output_dir, f'{base_filename}_statistics.csv')
    stats_df.to_csv(stats_output_path, index=False)

def compute_ttest(stained_patches_df):
    set1 = stained_patches_df[(stained_patches_df['am']) & (~stained_patches_df['tumor'])]['prob_lost_expression']
    set2 = stained_patches_df[(stained_patches_df['am']) & (stained_patches_df['tumor'])]['prob_lost_expression']

    set1 = np.array(set1)
    set2 = np.array(set2)

    # Perform t-test
    t_stat, p_value = ttest_ind(set1, set2, equal_var=False)

    print(f"T-test results:\n t-statistic: {t_stat}, p-value: {p_value} for slide {stained_patches_df['slide'].iloc[0]}")

    return p_value  # Return this directly


def ai_patch_prediction(staining, image_dir, stained_patches_df):
    filename = None
    # Load model
    model = load_model(staining)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # CommentToMarc: This is done so we only process files with AM and LO staining
    # Prepare valid file paths
    valid_files = stained_patches_df['img_path'].tolist()

    # Prepare dataset and dataloader
    infer_dataset = CustomImageDataset(image_dir, valid_files, transform=get_inference_augmentation())
    infer_loader = DataLoader(infer_dataset, batch_size=100, shuffle=False)

    predictions = []
    prediction_scores_list = []

    with torch.no_grad():
        for images, paths in tqdm(infer_loader, desc='Inferring patches with AI model'):
            images = images.to(device)
            outputs = model(images)

            for i in range(images.size(0)):
                prediction_scores = outputs[i].cpu().numpy()  # Raw scores for both classes
                prob_lost_expression = prediction_scores[0]  # Probability for "Lost expression"
                prob_maintained_expression = prediction_scores[1]  # Probability for "Maintained expression"
                if staining == 'lo':
                    # CommentToMarc
                    # 0.68 threshold obtained from calibrating with the LabelStudio validation data
                    prediction = 'Lost expression' if prob_lost_expression > 0.68 else 'Maintained expression'
                else:
                    prediction = 'Lost expression' if prob_lost_expression > prob_maintained_expression else 'Maintained expression'
                
                filename = paths[i].split('/')[-1]

                # Update the DataFrame with the probability of "Lost expression"
                idx = stained_patches_df[stained_patches_df['img_path'] == paths[i]].index

                if not idx.empty:
                    idx = idx[0]
                    stained_patches_df.at[idx, 'prediction'] = prediction
                    stained_patches_df.at[idx, 'prob_lost_expression'] = prob_lost_expression

                predictions.append({"filename": filename, "prediction": prediction})
                prediction_scores_list.append({
                    "filename": filename,
                    "lost_expression_prob": prob_lost_expression,
                    "maintained_expression_prob": prob_maintained_expression
                })



    # CommentToMarc
    # Save patches prediction scores to a CSV file and get statistics
    save_prediction_scores(prediction_scores_list, staining, image_dir, filename)
    stats = compute_prediction_statistics(prediction_scores_list)

    # Compute and print the t-test results as we saw that the Patch CNN did not work for AM staining
    p_value = 'Not applicable'
    overall_prediction = "Lost expression" if any(pred['prediction'] == "Lost expression" for pred in predictions) else "Maintained expression"
    predictions_df = pd.DataFrame(predictions)

    return overall_prediction, p_value, predictions_df, stats

