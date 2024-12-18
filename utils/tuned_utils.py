import optuna
import torch.optim as optim
import torch

def objective(trial, model, train_loader, val_loader, criterion, device, save_path=None):
    """
    Objective function for Optuna hyperparameter tuning.

    Args:
        trial: Optuna trial object.
        model: PyTorch model to optimize.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device for training (e.g., "cuda" or "cpu").
        save_path: Path to save the best model.

    Returns:
        float: Validation F1-score of the current trial.
    """
    # Hyperparameter suggestions
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout_prob = trial.suggest_float("dropout_prob", 0.2, 0.5)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW"])

    # Optimizer setup
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5, verbose=True)

    # Train the model
    from train_utils import train_model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_path=save_path,
        early_stop_patience=5,
        monitor_metric="val_f1",
        max_epochs=50
    )

    # Return validation F1-score
    return history["val_f1"][-1]


def run_study(model_fn, train_loader, val_loader, criterion, device, n_trials=20, save_path=None):
    """
    Runs an Optuna study to find the best hyperparameters.

    Args:
        model_fn: Function to create a model instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device for training (e.g., "cuda" or "cpu").
        n_trials: Number of trials to run.
        save_path: Path to save the best model.

    Returns:
        optuna.study.Study: The completed study.
    """
    def wrapped_objective(trial):
        model = model_fn(dropout_prob=trial.suggest_float("dropout_prob", 0.2, 0.5))
        return objective(trial, model, train_loader, val_loader, criterion, device, save_path)

    study = optuna.create_study(direction="maximize")
    study.optimize(wrapped_objective, n_trials=n_trials)

    print(f"[INFO] Best parameters: {study.best_params}")
    print(f"[INFO] Best F1-score: {study.best_value:.4f}")

    return study

# Example usage
if __name__ == "__main__":
    import torch.nn as nn
    from model_utils import initialize_model

    # Example: Define model function
    def model_fn(dropout_prob):
        return initialize_model("resnet18", num_classes=2, pretrained=True, dropout_prob=dropout_prob)

    # Define dummy dataloaders and criterion
    train_loader = val_loader = None  # Replace with actual DataLoaders
    criterion = nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run study
    study = run_study(model_fn, train_loader, val_loader, criterion, device, n_trials=20, save_path="best_model.pth")
