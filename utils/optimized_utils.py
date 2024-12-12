from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_optimizer_scheduler(model, lr=0.001, weight_decay=1e-4, scheduler_type="cosine", T_max=10):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    return optimizer, scheduler
