from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_datasets(data_dir, batch_size=32, num_workers=4):
    """
    Load train, validation, and test datasets with data augmentation for training.

    Args:
        data_dir (str): Root directory containing 'Train', 'Val', and 'Test' folders.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        tuple: train_loader, val_loader, test_loader, class_names
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),       # Random horizontal flipping
        transforms.RandomRotation(15),          # Random rotation
        transforms.RandomResizedCrop(224),      # Random crop and resize
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
    ])

    # Standard transformations for validation and test sets
    val_test_transform = transforms.Compose([
        transforms.Resize(256),                  # Resize to 256 for consistent input
        transforms.CenterCrop(224),             # Center crop to 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/Train", transform=train_transform)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/Val", transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/Test", transform=val_test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_dataset.classes
