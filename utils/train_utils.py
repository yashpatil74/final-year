import torch
import os
from tqdm import tqdm
from sklearn.metrics import recall_score, f1_score

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    device="cpu",
    save_path=None,
    early_stop_patience=5,
    monitor_metric="val_recall",  # Options: "val_recall", "val_loss", "val_f1"
    max_epochs=float("inf")
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

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss, val_correct = 0, 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

                preds = outputs.argmax(1).cpu().numpy()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        val_recall = recall_score(y_true, y_pred, average="binary")
        val_f1 = f1_score(y_true, y_pred, average="binary")

        # Update metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)

        # Print epoch summary
        print(
            f"Epoch [{epoch}]: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}"
        )

        # Early Stopping Logic
        current_metric = locals()[monitor_metric]  # Dynamically access the metric
        if (monitor_metric == "val_loss" and current_metric < best_metric) or \
           (monitor_metric != "val_loss" and current_metric > best_metric):
            best_metric = current_metric
            patience = 0
            # Save the best model
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"[INFO] Best model saved with {monitor_metric}: {best_metric:.4f}")
        else:
            patience += 1
            print(f"[INFO] No improvement in {monitor_metric}. Patience: {patience}/{early_stop_patience}")

        # Scheduler Step
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_metric)  # Monitor metric for ReduceLROnPlateau
            else:
                scheduler.step()

            # Log learning rate
            lr = optimizer.param_groups[0]['lr']
            print(f"[INFO] Learning rate adjusted to: {lr:.6f}")

    print(f"[INFO] Training stopped after {epoch} epochs.\n")
    return history
