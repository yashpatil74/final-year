from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import torch

def evaluate_model(model, test_loader, classes, device):
    model.eval()
    model.to(device)
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.argmax(1).cpu().numpy())

    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred, average="weighted", multi_class="ovr")

    return {"classification_report": report, "confusion_matrix": cm, "roc_auc": roc_auc}
