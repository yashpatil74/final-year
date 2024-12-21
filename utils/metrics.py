from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.transforms import ToPILImage


def evaluate_model(model, test_loader, classes, device, model_name="model", save_base_path="outputs/logs"):
    """
    Evaluates the model on the test dataset and returns key metrics. Saves evaluation results to a JSON file.

    Args:
        model: PyTorch model to evaluate.
        test_loader: DataLoader for test data.
        classes: List of class labels.
        device: Device to run evaluation on ("cuda" or "cpu").
        model_name: Name of the model for dynamic directory and file naming.
        save_base_path: Base directory to save evaluation logs.

    Returns:
        dict: A dictionary containing classification report, confusion matrix, ROC AUC scores, and true/predicted labels.
    """
    model.eval()
    model.to(device)
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            y_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.argmax(1).cpu().numpy())

    # Compute metrics
    classification_report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    confusion_mat = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, np.array(y_probs)[:, 1], average="weighted")

    # Convert results to JSON-serializable types
    results = {
        "classification_report": classification_report_dict,
        "confusion_matrix": confusion_mat.tolist(),
        "roc_auc": float(roc_auc),
        "y_true": list(map(int, y_true)),
        "y_pred": list(map(int, y_pred)),
        "y_scores": np.array(y_probs)[:, 1].tolist()
    }

    # Prepare dynamic directory and file path
    model_logs_path = os.path.join(save_base_path, model_name)
    os.makedirs(model_logs_path, exist_ok=True)
    file_name = os.path.join(model_logs_path, f"{model_name}_evaluation.json")

    # Save metrics to a JSON file
    with open(file_name, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[INFO] Evaluation metrics saved to {file_name}")

    return results


def plot_confusion_matrix(cm, classes, output_path=None, normalize=False):
    """
    Plots a confusion matrix.

    Args:
        cm: Confusion matrix.
        classes: List of class labels.
        output_path: Path to save the plot. If None, displays the plot.
        normalize: Whether to normalize the confusion matrix.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if output_path:
        plt.savefig(output_path)
        print(f"[INFO] Confusion matrix saved to {output_path}")
    else:
        plt.show()


def log_misclassified_samples(test_loader, y_true, y_pred, classes, model_name="model", save_base_path="outputs/logs"):
    """
    Logs misclassified samples for further analysis.

    Args:
        test_loader: DataLoader for the test dataset.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        classes: List of class labels.
        model_name: Name of the model for dynamic directory and file naming.
        save_base_path: Base directory to save logs.
    """
    misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred]
    print(f"[INFO] Number of misclassified samples: {len(misclassified_indices)}")

    model_logs_path = os.path.join(save_base_path, model_name)
    os.makedirs(model_logs_path, exist_ok=True)
    file_name = os.path.join(model_logs_path, f"{model_name}_misclassified_samples.json")

    misclassified_samples = [
        {"index": idx, "true_label": classes[true], "predicted_label": classes[pred]}
        for idx, (true, pred) in enumerate(zip(y_true, y_pred))
        if true != pred
    ]
    with open(file_name, "w") as f:
        json.dump(misclassified_samples, f, indent=4)
    print(f"[INFO] Misclassified samples saved to {file_name}")


def visualize_misclassified_samples(test_loader, y_true, y_pred, classes, num_samples=5):
    """
    Visualizes a few misclassified samples.

    Args:
        test_loader: DataLoader for the test dataset.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        classes: List of class labels.
        num_samples: Number of samples to visualize.
    """
    misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred]
    to_pil = ToPILImage()

    for idx in misclassified_indices[:num_samples]:
        image, label = test_loader.dataset[idx]
        plt.imshow(to_pil(image))
        plt.title(f"True: {classes[label]}, Pred: {classes[y_pred[idx]]}")
        plt.axis("off")
        plt.show()
