from albumentations import Compose, Resize, Normalize, HorizontalFlip, RandomRotate90, ColorJitter
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
        transform = Compose([
            Resize(224, 224),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    elif augmentation == "augmented":
        transform = Compose([
            Resize(224, 224),
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.2),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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
