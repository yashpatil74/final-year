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

# Load and preprocess the image
image_path = r"istockphoto-1224965512-612x612.jpg"
image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():  # Disable gradient calculation for inference
    outputs = model(input_tensor)  # Forward pass through the model
    _, predicted_class = torch.max(outputs, 1)  # Get the predicted class index

# Map predicted index to class labels
class_labels = ['fire', 'nofire']  # Replace with your class labels
predicted_label = class_labels[predicted_class.item()]  # Get the predicted label
print(f"Predicted Label: {predicted_label}")
