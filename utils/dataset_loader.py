from albumentations import (
    Compose, Resize, Normalize, HorizontalFlip, RandomRotate90, ColorJitter, GaussianBlur,
    ShiftScaleRotate, GridDistortion, CoarseDropout, GaussNoise
)
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np


class AlbumentationsDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root)
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image=np.array(image))["image"]
        return image, target


def load_datasets(data_dir, batch_size, augmentation="baseline"):
    if augmentation == "baseline":
        # Basic transformation: Resize and Normalize
        transform = Compose([
            Resize(224, 224),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    elif augmentation == "augmented":
        # Advanced augmentation: Adding distortions, noise, and dropout
        transform = Compose([
            Resize(224, 224),
            HorizontalFlip(p=0.5),  # Random horizontal flip
            RandomRotate90(p=0.5),  # Random rotations in multiples of 90 degrees
            ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),  # Shifts, scaling, rotations
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),  # Color jittering
            GaussianBlur(blur_limit=3, p=0.3),  # Slight blur
            GridDistortion(p=0.3),  # Grid-based distortion
            CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),  # Randomly drop sections
            GaussNoise(var_limit=(10.0, 50.0), p=0.2),  # Add Gaussian noise
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Standard normalization
            ToTensorV2()
        ])
    else:
        raise ValueError(f"Unsupported augmentation type: {augmentation}")

    # Load datasets
    train_dataset = AlbumentationsDataset(root=f"{data_dir}/train", transform=transform)
    val_dataset = AlbumentationsDataset(root=f"{data_dir}/val", transform=transform)
    test_dataset = AlbumentationsDataset(root=f"{data_dir}/test", transform=transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
