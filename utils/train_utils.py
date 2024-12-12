import torch
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    """
    Trains a PyTorch model and tracks training/validation metrics.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        num_epochs (int): Number of training epochs.
        device: Device to train on (e.g., "cuda" or "cpu").

    Returns:
        history (dict): Training and validation history.
    """
    model.to(device)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\nStarting training...\n")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_correct = 0, 0

        # Progress bar for training
        train_progress = tqdm(
            enumerate(train_loader),
            desc=f"Epoch [{epoch + 1}/{num_epochs}] - Training",
            total=len(train_loader),
            leave=False
        )

        for batch_idx, (images, labels) in train_progress:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()

            # Update progress bar with current batch loss
            train_progress.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        # Append metrics to history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Scheduler step
        if scheduler:
            scheduler.step()

        # Print epoch summary
        print(
            f"Epoch [{epoch + 1}/{num_epochs}]:\n"
            f"    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n"
            f"    Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}\n"
            f"    Learning Rate: {scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']:.6f}\n"
        )

    print("\nTraining complete!\n")
    return history
