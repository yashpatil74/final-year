import torch.nn as nn
from torchvision import models

def initialize_model(model_name, num_classes, pretrained=True, freeze_all=False, unfreeze_last_n=0):
    """
    Initializes the model for fine-tuning.

    Args:
        model_name (str): Model name (e.g., "resnet18", "vgg16", "efficientnet_b0").
        num_classes (int): Number of classes for classification.
        pretrained (bool): Use pretrained weights if True.
        freeze_all (bool): Freeze all layers if True.
        unfreeze_last_n (int): Number of last layers to unfreeze.

    Returns:
        model: PyTorch model.
    """
    if model_name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(pretrained=pretrained)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    # Freeze all feature extractor layers
    if freeze_all:
        for param in model.parameters():
            param.requires_grad = False

    # Ensure classifier layers remain trainable
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True

    # Optionally unfreeze last `n` layers
    if unfreeze_last_n > 0:
        layers = list(model.children())
        for layer in layers[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

    return model
