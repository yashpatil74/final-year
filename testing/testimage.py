import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torchvision.models import (
    ConvNeXt_Tiny_Weights, MobileNet_V2_Weights, EfficientNet_B0_Weights, ResNet18_Weights
)

def initialize_model(model_name, num_classes, pretrained=True, dropout_prob=0.5):
    if model_name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(model.classifier[2].in_features, num_classes),
        )

    elif model_name == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )

    elif model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )

    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(num_ftrs, num_classes)
        )

    else:
        raise ValueError(f"Model {model_name} not supported.")

    return model

# Define class labels
class_labels = ['fire', 'nofire']
num_classes = len(class_labels)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prompt user for image path
image_path = input("Enter the full path to the image: ")
if not os.path.exists(image_path):
    print("Image path does not exist. Please check and try again.")
    exit()

# Define model names and test types
models_to_test = ["convnext_tiny", "mobilenet_v2", "efficientnet_b0", "resnet18"]
test_types = ["baseline", "augmented", "tuned"]

# Load and preprocess the image
try:
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# Iterate through models and test types
results = []
for model_name in models_to_test:
    for test_type in test_types:
        print(f"Evaluating {model_name} ({test_type})")

        # Initialize the model
        model = initialize_model(model_name, num_classes, pretrained=False)

        # Load the trained weights
        try:
            state_dict_path = f"../outputs/models/{test_type}/{model_name}.pth"
            state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()  # Set the model to evaluation mode
        except Exception as e:
            print(f"Error loading weights for {model_name} ({test_type}): {e}")
            continue

        # Perform inference
        try:
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted_class = torch.max(outputs, 1)
                predicted_label = class_labels[predicted_class.item()]
                results.append({
                    "model": model_name,
                    "test_type": test_type,
                    "predicted_label": predicted_label
                })
        except Exception as e:
            print(f"Error during inference for {model_name} ({test_type}): {e}")

# Display results
print("\nResults:")
for result in results:
    print(f"\nModel: {result['model']}, Test Type: {result['test_type']}, Predicted Label: {result['predicted_label']}")
