import torch
from tqdm import tqdm
from sklearn.metrics import recall_score, f1_score, precision_score

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    device="cpu",
    save_path=None,
    early_stop_patience=8,
    monitor_metric="val_f1",  # Options: "val_recall", "val_loss", "val_f1"
    max_epochs=100
):
    """
    Trains a PyTorch model with early stopping, learning rate adjustment, and metric tracking.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler. Supports ReduceLROnPlateau or other schedulers.
        device: Device to train on (e.g., "cuda" or "cpu").
        save_path: Path to save the best model.
        early_stop_patience: Number of epochs to wait for improvement before stopping.
        monitor_metric: Metric to monitor for early stopping ("val_recall", "val_loss", "val_f1").
        max_epochs: Maximum allowable epochs.

    Returns:
        history (dict): Training and validation metrics history.
    """
    model.to(device)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_recall": [], "val_f1": []}

    # Create save directory if not exists
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Initialization
    best_metric = float("inf") if monitor_metric == "val_loss" else -float("inf")
    patience = 0
    epoch = 0

    print("\nStarting training...\n")
    while patience < early_stop_patience and epoch < max_epochs:
        epoch += 1

        # Training phase
        model.train()
        train_loss, train_correct = 0, 0
        y_train_true, y_train_pred = [], []

        train_progress = tqdm(train_loader, desc=f"Epoch [{epoch}] - Training", leave=False)
        for images, labels in train_progress:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            y_train_true.extend(labels.cpu().numpy())
            y_train_pred.extend(outputs.argmax(1).cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        train_f1 = f1_score(y_train_true, y_train_pred, average="binary")

        # Validation phase
        model.eval()
        val_loss, val_correct = 0, 0
        y_val_true, y_val_pred = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                y_val_true.extend(labels.cpu().numpy())
                y_val_pred.extend(outputs.argmax(1).cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        val_f1 = f1_score(y_val_true, y_val_pred, average="binary")
        val_recall = recall_score(y_val_true, y_val_pred, average="binary")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_recall"].append(val_recall)

        # Scheduler Step
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_f1 if monitor_metric == "val_f1" else val_recall)
            else:
                scheduler.step()

            lr = optimizer.param_groups[0]['lr']
            print(f"[INFO] Learning rate adjusted to: {lr:.6f}")

        # Early Stopping Logic
        current_metric = locals()[monitor_metric]  # Dynamically access the metric
        if (monitor_metric == "val_loss" and current_metric < best_metric) or \
           (monitor_metric != "val_loss" and current_metric > best_metric):
            best_metric = current_metric
            patience = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"[INFO] Best model saved with {monitor_metric}: {best_metric:.4f}")
        else:
            patience += 1
            print(f"[INFO] No improvement in {monitor_metric}. Patience: {patience}/{early_stop_patience}")

        # Print epoch summary
        print(
            f"Epoch [{epoch}]: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}"
        )

    print(f"[INFO] Training stopped after {epoch} epochs. Best {monitor_metric}: {best_metric:.4f}\n")
    return history
