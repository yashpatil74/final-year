import albumentations as A
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
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
    elif augmentation == "augmented":
        # Advanced augmentation: Adding distortions, noise, and dropout
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.Blur(blur_limit=7, p=0.3),
            A.MotionBlur(blur_limit=7, p=0.3),
            A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=64, max_width=64, p=0.5),
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.08, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        raise ValueError(f"Unsupported augmentation type: {augmentation}")

    train_dataset = AlbumentationsDataset(root=f"{data_dir}/train", transform=transform)
    val_dataset = AlbumentationsDataset(root=f"{data_dir}/val", transform=transform)
    test_dataset = AlbumentationsDataset(root=f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
