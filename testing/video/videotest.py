import torch
import cv2
import time
import numpy as np
from torchvision import models
from torchvision import transforms
from torch import nn
from PIL import Image

def initialize_model(model_name, num_classes, pretrained=True, freeze_all=False, unfreeze_last_n=0, dropout_prob=0.5):
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(num_ftrs, num_classes)
        )

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )

    elif model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )

    elif model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(model.classifier[2].in_features, num_classes),
        )

    else:
        raise ValueError(f"Model {model_name} not supported.")

    if freeze_all:
        for param in model.parameters():
            param.requires_grad = False

    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True

    if unfreeze_last_n > 0:
        layers = list(model.children())
        for layer in layers[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

    return model

MODEL_PATH_TEMPLATE = "../../outputs/models/{test}/{model_name}.pth"
video_path = input("Enter the path to the video file: ")
test_type = input("Select the test you want to try:\n1) baseline\n2) augmented\n3) tuned\n")
test_map = {
    "1": "baseline",
    "2": "augmented",
    "3": "tuned"
}
if test_type not in test_map:
    raise ValueError("Invalid test selection.")
test_type = test_map[test_type]
model_name = input("Select the model architecture:\n1) resnet18\n2) efficientnet_b0\n3) mobilenet_v2\n4) convnext_tiny\n")
model_map = {
    "1": "resnet18",
    "2": "efficientnet_b0",
    "3": "mobilenet_v2",
    "4": "convnext_tiny"
}
if model_name not in model_map:
    raise ValueError("Invalid model selection.")
model_name = model_map[model_name]
model_path = MODEL_PATH_TEMPLATE.format(test=test_type, model_name=model_name)
model = initialize_model(model_name, num_classes=2, pretrained=True, dropout_prob=0.5)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
cap = cv2.VideoCapture(video_path)
SAVE_OUTPUT = True
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
SKIP_FRAMES = 1
try:
    frame_number = 0
    processed_frame_count = 0
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or no input detected.")
            break
        frame_number += 1
        if frame_number % SKIP_FRAMES != 0:
            continue
        processed_frame_count += 1
        preprocessed_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        input_tensor = preprocess(preprocessed_frame).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item()
        confidence_threshold = 0.6
        if confidence < confidence_threshold:
            label = "Uncertain"
        else:
            label = "Fire" if prediction == 0 else "No Fire"
        fps = processed_frame_count / (time.time() - start_time)
        print(f"Processed Frame {processed_frame_count} (Video Frame {frame_number}): Prediction={label}, Confidence={confidence:.2f}, FPS={fps:.2f}")
        cv2.putText(frame, f'Test: {test_type} | Model: {model_name}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'{label} | Confidence: {confidence:.2f}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'Frame: {frame_number}', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Fire Detection", frame)
        if SAVE_OUTPUT:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
    if SAVE_OUTPUT:
        out.release()
    cv2.destroyAllWindows()
