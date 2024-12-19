from albumentations import (
    Compose, Resize, Normalize, HorizontalFlip, RandomRotate90, ShiftScaleRotate, ColorJitter,
    GaussianBlur, GaussNoise, GridDistortion, RandomFog, RandomRain, RandomShadow,
    CoarseDropout, Cutout
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
        image = Image.open(path).convert("RGB")  # Ensure RGB conversion
        if self.transform:
            image = self.transform(image=np.array(image))["image"]  # Apply augmentation
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
            Resize(224, 224),  # Uniform image resizing
            HorizontalFlip(p=0.5),  # Random horizontal flip
            RandomRotate90(p=0.5),  # Random 90-degree rotations
            ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),  # Random transformations
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),  # Simulate color changes
            GaussianBlur(blur_limit=5, p=0.3),  # Simulate slight blur
            GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Add Gaussian noise
            GridDistortion(p=0.3),  # Grid-based distortions for variation
            RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, alpha_coef=0.1, p=0.3),  # Simulate fog
            RandomRain(blur_value=3, drop_length=5, drop_width=1, p=0.3),  # Simulate rain
            RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=4, p=0.3),  # Add shadows
            CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),  # Random pixel drop
            Cutout(num_holes=4, max_h_size=32, max_w_size=32, fill_value=0, p=0.3),  # Remove small patches
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize for pre-trained models
            ToTensorV2()  # Convert to PyTorch Tensor
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


# Example usage
if __name__ == "__main__":
    # Directory containing train, val, test subfolders
    data_dir = "/path/to/data"
    batch_size = 32

    # Load datasets with advanced augmentation
    train_loader, val_loader, test_loader = load_datasets(data_dir, batch_size, augmentation="augmented")

    # Example: Display some augmented images from the train_loader
    import matplotlib.pyplot as plt
    for images, labels in train_loader:
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for img, ax in zip(images[:4], axes):  # Show the first 4 images
            ax.imshow(img.permute(1, 2, 0).numpy())  # Convert Tensor to NumPy for plotting
            ax.axis("off")
        plt.show()
        break