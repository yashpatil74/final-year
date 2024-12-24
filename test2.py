import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import convnext_tiny

# Define the model architecture
model = convnext_tiny(weights=None)  # Load ConvNeXt-Tiny without pretrained weights

# Redefine the classifier to match your training setup
model.classifier = nn.Sequential(
    nn.Flatten(),  # Flatten the input
    nn.Dropout(p=0.5),  # Adjust dropout probability if it was different during training
    nn.Linear(model.classifier[2].in_features, 2),  # Map features to 2 classes
)

# Load the trained weights
state_dict = torch.load(r"outputs\models\tuned\convnext_tiny_trial_best.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()  # Set the model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to model's expected input size
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Path to test directory
test_dir = r"wildfire_dataset_scaled/test"

# Class labels
class_labels = ['fire', 'nofire']  # Replace with your class labels

# Initialize counters for summary
total_images = 0
correct_predictions = 0
class_counts = {label: 0 for label in class_labels}
class_correct = {label: 0 for label in class_labels}

# Process each image in the test directory
for class_name in os.listdir(test_dir):  # Iterate over 'fire' and 'nofire' directories
    class_path = os.path.join(test_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    for image_name in os.listdir(class_path):  # Iterate over each image
        image_path = os.path.join(class_path, image_name)
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
            input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

            # Perform inference
            with torch.no_grad():  # Disable gradient calculation for inference
                outputs = model(input_tensor)  # Forward pass through the model
                _, predicted_class = torch.max(outputs, 1)  # Get the predicted class index

            predicted_label = class_labels[predicted_class.item()]  # Get the predicted label
            
            # Update counters
            total_images += 1
            class_counts[class_name] += 1
            if predicted_label == class_name:
                correct_predictions += 1
                class_correct[class_name] += 1

            # Print prediction
            print(f"Image: {image_name}, Actual: {class_name}, Predicted: {predicted_label}")

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

# Summary of results
print("\nSummary:")
print(f"Total Images: {total_images}")
print(f"Overall Accuracy: {correct_predictions / total_images * 100:.2f}%")

for label in class_labels:
    if class_counts[label] > 0:
        accuracy = class_correct[label] / class_counts[label] * 100
    else:
        accuracy = 0
    print(f"Class: {label}, Total: {class_counts[label]}, Correct: {class_correct[label]}, Accuracy: {accuracy:.2f}%")
