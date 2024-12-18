from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import torch
import numpy as np

def evaluate_model(model, test_loader, classes, device):
    """
    Evaluates the model on the test dataset and returns key metrics.

    Args:
        model: PyTorch model to evaluate.
        test_loader: DataLoader for test data.
        classes: List of class labels.
        device: Device to run evaluation on ("cuda" or "cpu").

    Returns:
        dict: A dictionary containing classification report, confusion matrix, and ROC AUC scores.
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
    roc_auc = roc_auc_score(y_true, np.array(y_probs)[:, 1], average="weighted", multi_class="ovr")

    return {
        "classification_report": classification_report_dict,
        "confusion_matrix": confusion_mat,
        "roc_auc": roc_auc
    }

def plot_confusion_matrix(cm, classes, normalize=False, cmap="Blues"):
    """
    Plots a confusion matrix.

    Args:
        cm: Confusion matrix.
        classes: List of class labels.
        normalize: Whether to normalize the values.
        cmap: Colormap to use.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.show()

def log_misclassified_samples(test_loader, y_true, y_pred, classes):
    """
    Logs misclassified samples for further analysis.

    Args:
        test_loader: DataLoader for the test dataset.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        classes: List of class labels.
    """
    misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred]
    print(f"[INFO] Number of misclassified samples: {len(misclassified_indices)}")

    # Optionally visualize a few misclassified samples
    import matplotlib.pyplot as plt
    from torchvision.transforms import ToPILImage

    to_pil = ToPILImage()
    for idx in misclassified_indices[:5]:
        image, label = test_loader.dataset[idx]
        plt.imshow(to_pil(image))
        plt.title(f"True: {classes[label]}, Pred: {classes[y_pred[idx]]}")
        plt.axis("off")
        plt.show()
