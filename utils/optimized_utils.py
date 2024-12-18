from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

def get_optimizer_scheduler(
    model,
    optimizer_type="AdamW",
    lr=0.001,
    weight_decay=1e-4,
    scheduler_type="cosine",
    T_max=10,
    patience=3,
    factor=0.5,
    step_size=10
):
    """
    Configures the optimizer and learning rate scheduler.

    Args:
        model: PyTorch model whose parameters will be optimized.
        optimizer_type (str): Type of optimizer ("AdamW", "SGD", "Adam").
        lr (float): Learning rate.
        weight_decay (float): Weight decay for regularization.
        scheduler_type (str): Type of scheduler ("cosine", "plateau", "step").
        T_max (int): Maximum number of iterations for CosineAnnealingLR.
        patience (int): Patience for ReduceLROnPlateau.
        factor (float): Factor by which the learning rate is reduced.
        step_size (int): Step size for StepLR.

    Returns:
        optimizer: Configured optimizer.
        scheduler: Configured learning rate scheduler.
    """
    # Initialize optimizer
    if optimizer_type == "AdamW":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == "Adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # Initialize scheduler
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=patience, factor=factor, verbose=True)
    elif scheduler_type == "step":
        scheduler = StepLR(optimizer, step_size=step_size, gamma=factor)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return optimizer, scheduler

# Example usage
if __name__ == "__main__":
    import torch.nn as nn

    # Dummy model for demonstration
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    model = DummyModel()

    optimizer, scheduler = get_optimizer_scheduler(
        model=model,
        optimizer_type="AdamW",
        lr=0.001,
        weight_decay=1e-4,
        scheduler_type="cosine",
        T_max=20
    )

    print("Optimizer and scheduler initialized successfully.")
