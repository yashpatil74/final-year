import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd
import torch


def plot_training(history, save_path="outputs/visualizations/training_validation_curves.png"):
    """
    Plot training and validation loss and accuracy over epochs.

    Args:
        history (dict): Contains train/val loss and accuracy.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', color='blue')
    plt.plot(history['val_acc'], label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(cm, classes, save_path="outputs/visualizations/confusion_matrix.png"):
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm (array): Confusion matrix.
        classes (list): Class labels.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.show()


def plot_roc_curve(model, test_loader, classes, device, save_path="outputs/visualizations/roc_curve.png"):
    """
    Plot ROC curve for multi-class classification.

    Args:
        model: PyTorch model.
        test_loader: DataLoader for test data.
        classes: List of class names.
        device: Device ('cuda' or 'cpu').
        save_path (str): Path to save the plot.
    """
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = label_binarize(all_labels, classes=range(len(classes)))

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_preds[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(save_path)
    plt.show()


def plot_precision_recall_curve(model, test_loader, classes, device, save_path="outputs/visualizations/precision_recall_curve.png"):
    """
    Plot precision-recall curve for multi-class classification.

    Args:
        model: PyTorch model.
        test_loader: DataLoader for test data.
        classes: List of class names.
        device: Device ('cuda' or 'cpu').
        save_path (str): Path to save the plot.
    """
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = label_binarize(all_labels, classes=range(len(classes)))

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(all_labels[:, i], all_preds[:, i])
        plt.plot(recall, precision, label=f"{class_name}")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(save_path)
    plt.show()


def plot_per_class_performance(report, classes, save_path="outputs/visualizations/per_class_performance.png"):
    """
    Plot per-class precision, recall, and F1-score as a bar chart.

    Args:
        report (dict): Classification report.
        classes (list): Class labels.
        save_path (str): Path to save the plot.
    """
    class_metrics = {cls: report[cls] for cls in classes}
    df = pd.DataFrame(class_metrics).T[['precision', 'recall', 'f1-score']]

    df.plot(kind='bar', figsize=(10, 6))
    plt.title('Per-Class Performance')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.savefig(save_path)
    plt.show()
