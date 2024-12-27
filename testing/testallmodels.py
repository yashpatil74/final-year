import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torchvision.models import (
    ConvNeXt_Tiny_Weights, MobileNet_V2_Weights, EfficientNet_B0_Weights, ResNet18_Weights
)
from tqdm import tqdm

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

print("Select the test type:")
test_types = ["baseline", "augmented", "tuned"]
for i, test_type in enumerate(test_types, 1):
    print(f"{i}. {test_type}")

test_choice = input("Enter the number corresponding to your choice: ")
try:
    test_type = test_types[int(test_choice) - 1]
except (ValueError, IndexError):
    print("Invalid choice. Defaulting to 'baseline'.")
    test_type = "baseline"

models_to_test = ["convnext_tiny", "mobilenet_v2", "efficientnet_b0", "resnet18"]

class_labels = ['fire', 'nofire']
num_classes = len(class_labels)

test_dir = r"../wildfire_dataset_scaled/test"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

final_summary = []

for model_name in models_to_test:
    print(f"\nTesting model: {model_name}")

    model = initialize_model(model_name, num_classes, pretrained=False)

    try:
        state_dict_path = f"../outputs/models/{test_type}/{model_name}.pth"
        state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()  # Set the model to evaluation mode
    except Exception as e:
        print(f"Error loading weights for {model_name}: {e}")
        continue

    total_images = 0
    correct_predictions = 0
    class_counts = {label: 0 for label in class_labels}
    class_correct = {label: 0 for label in class_labels}
    total_inference_time = 0  # Total time for inference

    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        image_list = os.listdir(class_path)

        # Add a progress bar for images
        print(f"Processing images for class '{class_name}'...")
        for image_name in tqdm(image_list, desc=f"Class: {class_name}"):
            image_path = os.path.join(class_path, image_name)
            try:
                image = Image.open(image_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0)

                start_time = time.time()
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted_class = torch.max(outputs, 1)
                end_time = time.time()

                inference_time = end_time - start_time
                total_inference_time += inference_time

                predicted_label = class_labels[predicted_class.item()]

                total_images += 1
                class_counts[class_name] += 1
                if predicted_label == class_name:
                    correct_predictions += 1
                    class_correct[class_name] += 1

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")

    avg_inference_time = total_inference_time / total_images if total_images > 0 else 0
    model_summary = {
        "model": model_name,
        "total_images": total_images,
        "overall_accuracy": correct_predictions / total_images * 100 if total_images > 0 else 0,
        "avg_inference_time": avg_inference_time,
        "class_wise": {
            label: {
                "total": class_counts[label],
                "correct": class_correct[label],
                "accuracy": class_correct[label] / class_counts[label] * 100 if class_counts[label] > 0 else 0
            } for label in class_labels
        }
    }
    final_summary.append(model_summary)

print("\nFinal Summary:")
for summary in final_summary:
    print(f"\nModel: {summary['model']}")
    print(f"Total Images: {summary['total_images']}")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.2f}%")
    print(f"Average Inference Time per Image: {summary['avg_inference_time']:.4f} seconds")
    for label, metrics in summary['class_wise'].items():
        print(f"Class: {label}, Total: {metrics['total']}, Correct: {metrics['correct']}, Accuracy: {metrics['accuracy']:.2f}%")
