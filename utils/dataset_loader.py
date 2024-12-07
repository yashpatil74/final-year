from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_datasets(data_dir, batch_size=32, num_workers=4):
    """
    Load train, validation, and test datasets using torchvision's ImageFolder.

    Args:
        data_dir (str): Root directory containing 'train', 'val', and 'test' folders.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        tuple: train_loader, val_loader, test_loader, class_names
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset.classes
