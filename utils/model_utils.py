import torch
import torch.nn as nn
from torchvision import models

def initialize_model(model_name, num_classes, pretrained=True):
    """
    Initialize a pretrained model with a modified final layer.

    Args:
        model_name (str): Model architecture (e.g., 'resnet18', 'efficientnet_b0').
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use pretrained weights.

    Returns:
        torch.nn.Module: Initialized model.
    """
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model

def save_model(model, path):
    """Save the model to the specified path."""
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    """Load a model's state dictionary from the specified path."""
    model.load_state_dict(torch.load(path, map_location=device))
    return model
