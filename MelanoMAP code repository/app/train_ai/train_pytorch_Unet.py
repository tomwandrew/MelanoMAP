import os
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
import argparse
import wandb
from utils.dataloaders import SegDataset
from utils.utils import (get_preprocessing_carlos, CategoricalFocalLoss, TotalLoss,
                          get_training_augmentation, get_validation_augmentation)

NUM_WORKERS = 4
DEFAULT_LR = 0.0001
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16

def get_dataloaders(x_train_dir: str, y_train_dir: str, x_valid_dir: str, y_valid_dir: str, 
                    preprocessing_fn, CLASSES: list, batch_size: int) -> tuple:
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, valid_loader

def train_epoch(model, train_epoch_runner, valid_epoch_runner, train_loader, valid_loader, 
                optimizer, max_score, args, epoch):
    print(f'\nEpoch: {epoch}')

    train_logs = train_epoch_runner.run(train_loader)
    valid_logs = valid_epoch_runner.run(valid_loader)
    
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, f'./best_model_{args.staining}_4.pth')
        print('Model saved!')

    wandb.log({
        'train_loss': train_logs['total loss'], 
        'train_iou_score': train_logs['iou_score'], 
        'train_fscore': train_logs['fscore'], 
        'valid_loss': valid_logs['total loss'],
        'valid_iou_score': valid_logs['iou_score'], 
        'valid_fscore': valid_logs['fscore']
    })

    if epoch == 180:
        optimizer.param_groups[0]['lr'] /= 2
        print('Decrease decoder learning rate to 1e-5!')

def setup(args):
    data_path = os.path.join('/data/20230414/processed', args.staining)
    x_train_dir = os.path.join(data_path, 'train', 'images')
    y_train_dir = os.path.join(data_path, 'train', 'joined')
    x_valid_dir = os.path.join(data_path, 'validation', 'images')
    y_valid_dir = os.path.join(data_path, 'validation', 'joined')

    CLASSES = args.classes
    BACKBONE = args.backbone
    LR = args.lr
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    activation = 'sigmoid' if len(CLASSES) == 1 else 'softmax'
    
    model = smp.UnetPlusPlus(
        encoder_name=BACKBONE, 
        classes=len(CLASSES), 
        activation=activation,
    )

    metrics = [utils.metrics.IoU(threshold=0.5), utils.metrics.Fscore(threshold=0.5)]

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(BACKBONE)
    train_loader, valid_loader = get_dataloaders(x_train_dir, y_train_dir, x_valid_dir, y_valid_dir, preprocessing_fn, CLASSES, BATCH_SIZE)
    
    dice_loss = utils.losses.DiceLoss()
    focal_loss = CategoricalFocalLoss()
    total_loss = TotalLoss(dice_loss, focal_loss)


    train_epoch_runner = utils.train.TrainEpoch(
        model, 
        loss=total_loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch_runner = utils.train.ValidEpoch(
        model, 
        loss=total_loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    wandb.init(project=f'{args.staining.upper()} segmentation', config=args)
    wandb.watch(model)

    max_score = 0

    for epoch in range(EPOCHS):
        train_epoch(model, train_epoch_runner, valid_epoch_runner, train_loader, valid_loader, optimizer, max_score, args, epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='UNET AMBLor Test')

    parser.add_argument('--staining', type=str, default='lo')
    parser.add_argument('--backbone', type=str, default='efficientnet-b0')
    parser.add_argument('--classes', type=str, nargs='+', default=['background', 'tissue', 'tumour', 'biomarker'])
    parser.add_argument('--lr', type=float, default=DEFAULT_LR)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)

    args = parser.parse_args()

    setup(args)