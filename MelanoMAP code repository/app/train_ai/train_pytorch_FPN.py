import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import functional as F
import segmentation_models_pytorch as smp

# Needed to load the loss and the metrics
from segmentation_models_pytorch import utils

from sklearn.metrics import jaccard_score
import numpy as np
from tqdm import tqdm

from utils.dataloaders import SegDataset
from utils.utils import (get_preprocessing_carlos, CategoricalFocalLoss, TotalLoss,
                          get_training_augmentation, get_simple_train_augmentation, get_validation_augmentation)
import argparse
import wandb

def get_dataloaders(x_train_dir, y_train_dir, x_valid_dir, y_valid_dir, preprocessing_fn, CLASSES):
    train_dataset = SegDataset(
        x_train_dir, 
        y_train_dir, 
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing_carlos(preprocessing_fn),
        classes=CLASSES,
    )

    valid_dataset = SegDataset(
        x_valid_dir, 
        y_valid_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing_carlos(preprocessing_fn),
        classes=CLASSES,
    )

    # Prepare DataLoader
    train_loader = DataLoader(train_dataset, batch_size=82, shuffle=True, num_workers=15)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=15)
    return train_loader, valid_loader


def run_train_epoch(model, train_epoch_runner, valid_epoch_runner, train_loader, valid_loader, optimizer, max_score, args, epoch):
    print('\nEpoch: {}'.format(epoch))

    train_logs = train_epoch_runner.run(train_loader)
    valid_logs = valid_epoch_runner.run(valid_loader)
    
    # Save model if validation IoU score improves
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, f'./best_model_{args.staining}_4.pth')
        print('Model saved!')

    # Log metrics to Weights & Biases
    wandb.log({
        'train_loss': train_logs['total loss'], 
        'train_iou_score': train_logs['iou_score'], 
        'train_fscore': train_logs['fscore'], 
        'valid_loss': valid_logs['total loss'],
        'valid_iou_score': valid_logs['iou_score'], 
        'valid_fscore': valid_logs['fscore']
    })

    # Decrease learning rate at epoch 180
    if epoch == 180:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 2
        print('Decrease decoder learning rate to 1e-5!')

    return max_score

def setup(args: argparse.Namespace):
    data_path = os.path.join('/data/20230414/processed', args.staining)
    x_train_dir = os.path.join(data_path, 'train', 'images')
    y_train_dir = os.path.join(data_path, 'train', 'masks')
    x_valid_dir = os.path.join(data_path, 'validation', 'images')
    y_valid_dir = os.path.join(data_path, 'validation', 'masks')

    CLASSES = args.classes
    BACKBONE = args.backbone
    LR = args.lr
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    activation = 'sigmoid' if len(CLASSES) == 1 else 'softmax'
    
    # Create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=BACKBONE, 
        classes=len(CLASSES), 
        activation=activation,
    )

    metrics = [utils.metrics.IoU(threshold=0.5), utils.metrics.Fscore(threshold=0.5)]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    preprocessing_fn = smp.encoders.get_preprocessing_fn(BACKBONE)
    train_loader, valid_loader = get_dataloaders(x_train_dir, y_train_dir, x_valid_dir, y_valid_dir, preprocessing_fn, CLASSES)
    
    # Create the loss function
    dice_loss = utils.losses.DiceLoss()
    focal_loss = CategoricalFocalLoss()
    total_loss = TotalLoss(dice_loss, focal_loss)

    # Create epoch runners
    train_epoch = utils.train.TrainEpoch(
        model, 
        loss=total_loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = utils.train.ValidEpoch(
        model, 
        loss=total_loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    # Initialize Weights & Biases
    wandb.init(project=f'{args.staining.upper()} segmentation', entity='catai', config=vars(args))
    wandb.watch(model)

    max_score = 0

    for i in range(EPOCHS):
        max_score = run_train_epoch(model, train_epoch, valid_epoch, train_loader, valid_loader, optimizer, max_score, args, i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='UNET AMBLor Test')

    parser.add_argument('--staining', type=str, default='am')
    parser.add_argument('--backbone', type=str, help='backbone', default='efficientnet-b0')
    parser.add_argument('--classes', type=str, help='classes', default=['background', 'tumour', 'am', 'lo'])
    parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    parser.add_argument('--epochs', type=int, help='epochs', default=100)
    parser.add_argument('--batch_size', type=int, help='batch size', default=128)

    args = parser.parse_args()
    setup(args)
