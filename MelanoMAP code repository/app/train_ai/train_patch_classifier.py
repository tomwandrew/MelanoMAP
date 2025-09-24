import os
import torch
from torchvision import datasets, models
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.utils.class_weight import compute_class_weight
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# Initialize WandB (make sure you have an account and have installed wandb library)
import wandb

from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# Paths to your train and validation data
DATA_DIR = '/data/tissue_n_background_marc/am'

#python train_patch_classifier.py

def get_training_augmentation():
    print("Getting training augmentations...")
    train_transform = [
        A.Resize(256, 256),  # Slightly larger resize
        A.RandomCrop(224, 224),  # Random crop to model input size
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),  # Add vertical flips
        A.Rotate(limit=45, p=0.5),  # Increase the rotation limit
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Brightness & contrast
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),  # Color shifts
        A.GaussNoise(var_limit=(10, 50), p=0.5),  # Gaussian noise
        A.CoarseDropout(
            max_holes=8,
            max_height=16,
            max_width=16,
            min_holes=1,
            fill_value=0,
            p=0.5
        ),  # Used as Cutout augmentation
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),  # Shift, Scale, and Rotate
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    print("Getting validation augmentations...")
    val_transform = [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(val_transform)

# Custom dataset to integrate albumentations with PyTorch ImageFolder
class CustomDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=datasets.folder.default_loader):
        super(CustomDataset, self).__init__(root, transform, target_transform, loader)
        self.augment = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.augment:
            image = self.augment(image=np.array(image))['image']
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target, path  # Return path for logging images to WandB


def get_class_weights(dataset):    
    print("Calculating class weights...")
    # Counting the number of samples per class
    class_counts = np.bincount([label for _, label, _ in dataset])
    # Computing class weights
    class_weights = 1. / class_counts
    # Assign a weight to each sample
    sample_weights = np.array([class_weights[label] for _, label, _ in dataset])
    return sample_weights



def calculate_metrics(y_true, y_pred, y_score=None, num_classes=None):
    print("Calculating metrics...")
    metrics = {
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    }

    if y_score is not None and num_classes is not None:
        y_true_binary = label_binarize(y_true, classes=range(num_classes))
        if num_classes == 2:  # ROC AUC for binary classification
            metrics['roc_auc'] = roc_auc_score(y_true_binary, y_score[:, 1])
        else:  # ROC AUC for multiclass
            metrics['roc_auc'] = roc_auc_score(y_true_binary, y_score, multi_class='ovr')

    return metrics

def validate(model, val_loader, criterion, device, num_classes):
    print("Starting validation...")
    model.eval()
    val_running_loss = 0.0
    y_true, y_pred, y_score = [], [], []
   
    progress_bar = tqdm(val_loader, desc='Validation', leave=False)
    with torch.no_grad():
        for inputs, labels, paths in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            scores, predictions = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            y_score.extend(outputs.cpu().detach().numpy())

            progress_bar.set_postfix({'val_loss': val_running_loss / (progress_bar.n + 1)})


    val_loss = val_running_loss / len(val_loader)
    metrics = calculate_metrics(y_true, y_pred, y_score=np.array(y_score), num_classes=num_classes)
    metrics['loss'] = val_loss

    print(f"Validation loss: {val_loss:.4f}")
    return metrics

def train_one_epoch(model, train_loader, criterion, optimizer, device, num_classes):
    print("Starting training for one epoch...")
    model.train()
    running_loss = 0.0
    y_true, y_pred, y_score = [], [], []

    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for inputs, labels, paths in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        scores, predictions = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())
        y_score.extend(outputs.cpu().detach().numpy())
        progress_bar.set_postfix({'train_loss': running_loss / (progress_bar.n + 1)})


    train_loss = running_loss / len(train_loader)
    metrics = calculate_metrics(y_true, y_pred, y_score=np.array(y_score), num_classes=num_classes)
    metrics['loss'] = train_loss

    print(f"Training loss: {train_loss:.4f}")
    return metrics


def train(args):
    print("Starting training process...")
    num_classes = len(args.classes)
    data_path = os.path.join(DATA_DIR, args.staining)
    data_path = DATA_DIR
    train_dir = os.path.join(data_path, 'train')
    valid_dir = os.path.join(data_path, 'validation')

    # Load the datasets with custom augmentations
    train_dataset = CustomDataset(train_dir, transform=get_training_augmentation())
    val_dataset = CustomDataset(valid_dir, transform=get_validation_augmentation())
    # Getting class weights for weighted sampling
    sample_weights = get_class_weights(train_dataset)
    weighted_sampler = WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=weighted_sampler, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # Step 4: Creating the EfficientNet-B0 model without pretrained weights
    # Load pre-trained ViT model
    model = models.efficientnet_v2_s(pretrained=True, dropout=0.3)
    
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(args.classes))

    #wandb.init(name= f'{args.staining.upper()} staining num classes {len(args.classes)} regularization', project="Patch classifier", entity='catai')
    wandb.init(name= f'Background and Tissue num classes {len(args.classes)} regularization')
    model.to(device)
     # Loss Function
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6, last_epoch=-1)

    best_balanced_acc = 0.0
    no_improvement_epochs = 0
    early_stopping_patience = 80  # or whatever number of epochs you find appropriate

    best_balanced_acc = 0.0
    history = {'train': [], 'val': []}
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, num_classes)
        val_metrics = validate(model, val_loader, criterion, device, num_classes)

        # Update learning rate
        scheduler.step()


        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # Retrieve current learning rate from optimizer
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics and learning rate to WandB
        log_data = {"epoch": epoch, "learning_rate": current_lr}
        for key in train_metrics:
            log_data[f"{key}/train"] = train_metrics[key]
        for key in val_metrics:
            log_data[f"{key}/val"] = val_metrics[key]

        # Log all metrics at once
        wandb.log(log_data)

        print(f'Validation Metrics: {val_metrics}')

        # Checkpoint model based on best balanced accuracy
        if val_metrics['balanced_accuracy'] > best_balanced_acc:
            best_balanced_acc = val_metrics['balanced_accuracy']
            best_state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_balanced_acc': best_balanced_acc,
                'history': history
            }
            torch.save(best_state, f'best_classifier_background_tissue_am.pth')
            print("Model checkpoint saved.")
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    wandb.finish()
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'EfficientNet B0 AMBLor Test',)

    parser.add_argument('--staining', type=str, default='am')
   # parser.add_argument('--classes', type=str, help='classes', default=['Maintained', 'Lost'])#, 'Unable to assess'])
    parser.add_argument('--classes', type=str, help='classes', default=['Background', 'Tissue'])
    parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    parser.add_argument('--epochs', type=int, help='epochs', default=500)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)


    args = parser.parse_args()
    pru = train(args)
    #train(pru)
