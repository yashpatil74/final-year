from albumentations import (
    Compose, Resize, Normalize, HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast, RandomFog
)
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image
import numpy as np
import os
import logging
import torch
import torch.nn as nn

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AlbumentationsDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root)
        self.transform = transform
        logging.info(f"Initialized AlbumentationsDataset with root: {root}")

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")  # Ensure RGB conversion
        if self.transform:
            image = self.transform(image=np.array(image))["image"]  # Apply augmentation
        return image, target

def compute_class_weights(data_dir, smoothing_factor=0.5):
    """
    Compute scaled and smoothed class weights for the dataset to handle imbalance.
    Args:
        data_dir (str): Path to dataset folder.
        smoothing_factor (float): Factor to reduce extremity of weights.

    Returns:
        weights (dict): Scaled and smoothed class weights.
    """
    logging.info(f"Computing class weights from directory: {data_dir}")
    counts = {"fire": 0, "nofire": 0}
    for cls in counts.keys():
        class_dir = os.path.join(data_dir, cls)
        counts[cls] = len(os.listdir(class_dir))
        logging.info(f"Class '{cls}' has {counts[cls]} samples.")
    total = sum(counts.values())
    raw_weights = {cls: total / count for cls, count in counts.items()}
    max_weight = max(raw_weights.values())
    scaled_weights = {cls: (weight / max_weight) ** smoothing_factor for cls, weight in raw_weights.items()}
    logging.info(f"Computed class weights: {scaled_weights}")
    return scaled_weights

def load_datasets(data_dir, batch_size, augmentation="baseline", weighted_sampler=True):
    """
    Enhanced data loader with class balancing options.

    Args:
        data_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for loaders.
        augmentation (str): Type of augmentation ("baseline" or "augmented").
        weighted_sampler (bool): Use a weighted sampler for class balancing.

    Returns:
        train_loader, val_loader, test_loader: DataLoaders for training, validation, and testing.
    """
    logging.info(f"Loading datasets from {data_dir} with augmentation type '{augmentation}'.")

    # Augmentation selection
    if augmentation == "baseline":
        logging.info("Applying baseline augmentations: Resize and Normalize.")
        transform = Compose([
            Resize(224, 224),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    elif augmentation == "augmented":
        logging.info("Applying selected augmentations for wildfire scenarios.")
        transform = Compose([
            Resize(224, 224),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0.02, scale_limit=0.02, rotate_limit=10, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        raise ValueError(f"Unsupported augmentation type: {augmentation}")

    # Load datasets
    train_dataset = AlbumentationsDataset(root=f"{data_dir}/train", transform=transform)
    val_dataset = AlbumentationsDataset(root=f"{data_dir}/val", transform=transform)
    test_dataset = AlbumentationsDataset(root=f"{data_dir}/test", transform=transform)

    logging.info("Datasets initialized. Preparing DataLoaders...")

    if weighted_sampler:
        logging.info("Using WeightedRandomSampler for class balancing.")
        class_weights = compute_class_weights(f"{data_dir}/train")
        weights = [class_weights[train_dataset.classes[label]] for _, label in train_dataset]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        logging.info("Using default sampler for training data.")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logging.info("DataLoaders created successfully.")
    return train_loader, val_loader, test_loader

# Define a weighted loss function to ensure balance during training
def get_loss_function(class_weights):
    class_weights_tensor = torch.tensor([class_weights['nofire'], class_weights['fire']], dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    logging.info("Initialized weighted CrossEntropyLoss function.")
    return loss_fn
