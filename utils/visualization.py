import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import numpy as np


def plot_training(history, output_path=None):
    """
    Plots training and validation loss and accuracy.

    Args:
        history (dict): Dictionary containing training and validation metrics.
        output_path (str): Path to save the plot. If None, displays the plot.
    """
    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    # Save or show the plot
    if output_path:
        plt.savefig(output_path)
        print(f"[INFO] Training plot saved to {output_path}")
    else:
        plt.show()


def plot_confusion_matrix(cm, classes, output_path=None, normalize=False):
    """
    Plots a confusion matrix.

    Args:
        cm (np.array): Confusion matrix.
        classes (list): List of class labels.
        output_path (str): Path to save the plot. If None, displays the plot.
        normalize (bool): Whether to normalize the confusion matrix.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Save or show the plot
    if output_path:
        plt.savefig(output_path)
        print(f"[INFO] Confusion matrix saved to {output_path}")
    else:
        plt.show()


def plot_precision_recall(y_true, y_scores, output_path=None):
    """
    Plots the Precision-Recall curve.

    Args:
        y_true (list): Ground truth labels.
        y_scores (list): Predicted probabilities for the positive class.
        output_path (str): Path to save the plot. If None, displays the plot.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)

    # Save or show the plot
    if output_path:
        plt.savefig(output_path)
        print(f"[INFO] Precision-Recall curve saved to {output_path}")
    else:
        plt.show()


def plot_roc_curve(y_true, y_scores, output_path=None):
    """
    Plots the ROC curve.

    Args:
        y_true (list): Ground truth labels.
        y_scores (list): Predicted probabilities for the positive class.
        output_path (str): Path to save the plot. If None, displays the plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker=".", label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.grid(True)

    # Save or show the plot
    if output_path:
        plt.savefig(output_path)
        print(f"[INFO] ROC curve saved to {output_path}")
    else:
        plt.show()
