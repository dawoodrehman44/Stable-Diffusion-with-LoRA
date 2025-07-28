import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    confusion_matrix, brier_score_loss
)

def compute_classwise_metrics(y_true, y_probs, class_names):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    num_classes = y_true.shape[1]

    metrics = {
        "auc": [],
        "precision": [],
        "recall": [],
        "tpr": [],
        "fpr": [],
        "ece": [],
        "bce": [],
        "error_rate": []
    }

    for i in range(num_classes):
        yt = y_true[:, i]
        yp = y_probs[:, i]
        yp_bin = (yp >= 0.5).astype(int)

        try:
            auc = roc_auc_score(yt, yp)
            precision = precision_score(yt, yp_bin, zero_division=0)
            recall = recall_score(yt, yp_bin, zero_division=0)  # Recall == TPR
            tn, fp, fn, tp = confusion_matrix(yt, yp_bin, labels=[0, 1]).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr = recall  # Just to make it explicit
            ece = brier_score_loss(yt, yp)
            bce = F.binary_cross_entropy(torch.tensor(yp), torch.tensor(yt)).item()
            err_rate = (fp + fn) / (tp + tn + fp + fn)
        except Exception:
            auc = precision = recall = fpr = tpr = ece = bce = err_rate = float('nan')

        metrics["auc"].append(auc)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["tpr"].append(tpr)
        metrics["fpr"].append(fpr)
        metrics["ece"].append(ece)
        metrics["bce"].append(bce)
        metrics["error_rate"].append(err_rate)

        print(f"{class_names[i]}: AUC={auc:.4f}, Precision={precision:.4f}, "
              f"TPR={tpr:.4f}, FPR={fpr:.4f}, ECE={ece:.4f}, "
              f"BCE={bce:.4f}, ErrorRate={err_rate:.4f}")

    return metrics

def average_metrics(metrics_dict):
    return {k: np.nanmean(v) for k, v in metrics_dict.items()}
