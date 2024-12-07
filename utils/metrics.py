from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
import numpy as np
import torch

def evaluate_model(model, test_loader, classes, device):
    """
    Evaluate the model on the test set and compute various metrics.

    Args:
        model: PyTorch model.
        test_loader: DataLoader for test data.
        classes: List of class names.
        device: Device ('cuda' or 'cpu').

    Prints:
        Classification report and confusion matrix.
    Returns:
        dict: Dictionary containing evaluation metrics (accuracy, precision, recall, F1-score, MCC, AUC).
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = report['accuracy']
    mcc = matthews_corrcoef(all_labels, all_preds)

    # Compute AUC (if multi-class, calculate macro-averaged AUC)
    all_labels_bin = label_binarize(all_labels, classes=range(len(classes)))
    auc = roc_auc_score(all_labels_bin, np.array(all_preds), average="macro", multi_class="ovr")

    # Print Metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"],
        "mcc": mcc,
        "auc": auc
    }
