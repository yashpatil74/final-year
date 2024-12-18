import torch.nn as nn
from torchvision import models
from torchvision.models import (
    VGG16_Weights, ResNet18_Weights, ResNet50_Weights,
    EfficientNet_B0_Weights, MobileNet_V2_Weights,
    DenseNet121_Weights, ConvNeXt_Tiny_Weights
)

def initialize_model(model_name, num_classes, pretrained=True, freeze_all=False, unfreeze_last_n=0, dropout_prob=0.5):
    """
    Initializes the model for fine-tuning with added Dropout layers to reduce overfitting.

    Args:
        model_name (str): Model name (e.g., "resnet18", "vgg16", "efficientnet_b0").
        num_classes (int): Number of classes for classification.
        pretrained (bool): Use pretrained weights if True.
        freeze_all (bool): Freeze all layers if True.
        unfreeze_last_n (int): Number of last layers to unfreeze.
        dropout_prob (float): Dropout probability to apply in the classifier head.

    Returns:
        model: PyTorch model.
    """
    # Select model and weights
    if model_name == "vgg16":
        weights = VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),  # Add Dropout here
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),  # Add another Dropout here
            nn.Linear(4096, num_classes)
        )

    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),  # Add Dropout before the FC layer
            nn.Linear(num_ftrs, num_classes)
        )

    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),  # Add Dropout here
            nn.Linear(num_ftrs, num_classes)
        )

    elif model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),  # Add Dropout here
            nn.Linear(model.classifier[1].in_features, num_classes)
        )

    elif model_name == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),  # Add Dropout here
            nn.Linear(model.classifier[1].in_features, num_classes)
        )

    elif model_name == "densenet121":
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),  # Add Dropout here
            nn.Linear(model.classifier.in_features, num_classes)
        )

    if model_name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        
        # Update the classifier with Flatten and Dropout
        model.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten input from [batch_size, 768, 1, 1] to [batch_size, 768]
            nn.Dropout(p=dropout_prob),  # Add Dropout for regularization
            nn.Linear(model.classifier[2].in_features, num_classes),  # Classifier head
        )
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
