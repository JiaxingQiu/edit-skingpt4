from typing import Any, Dict, Optional, Sequence
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


def eval_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    average: str = "macro",
    labels: Optional[Sequence[str]] = None,
    return_report: bool = False,
    return_confusion: bool = False,
) -> Dict[str, Any]:
    """Compute evaluation metrics for string labels.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        average: Averaging strategy for precision/recall/F1 (e.g., "macro", "micro", "weighted").
        labels: Optional label set to control ordering for report/confusion matrix.
        return_report: If True, include a classification report string.
        return_confusion: If True, include a confusion matrix (as a numpy array).

    Returns:
        A dictionary with keys: accuracy, precision, recall, f1, and optionally report, confusion.
    """

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, labels=labels, zero_division=0
    )

    result: Dict[str, Any] = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    if return_report:
        result["report"] = classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        )

    if return_confusion:
        result["confusion"] = confusion_matrix(y_true, y_pred, labels=labels)

    return result


from tqdm import tqdm
import pandas as pd
from model_skingpt4 import *
def _normalize(s: str, target: str) -> str:
    if target == "y3":
        out = str(s).strip().lower()
        if "malignant" in out:
            return "malignant"
        elif "benign" in out:
            return "benign"
        elif "other" in out:
            return "other"
        else:
            return "unknown"

def eval_ft_skingpt4(chat, dataset, temperature=0.1, 
                     target="y3", 
                     question="Is the lesion malignant or benign, or other?"):
    # model, vis_processor, chat = init_chat()
    labels = sorted({_normalize(str(dataset[i]['y'][target]), target) for i in range(len(dataset))})
    rows = []
    y_true, y_pred = [], []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        img = sample['image']
        gt = _normalize(sample['y'][target], target)
        pred = _normalize(chat_with_image(chat, img, question, temperature=temperature), target)
        y_true.append(gt)
        y_pred.append(pred)
        rows.append({
            "gt": _normalize(sample['y'][target], target),
            "pred": _normalize(pred, target),
            "id_patient": sample['id']['id_patient'],
            "id_filename": sample['id']['id_filename'],
        })
    metrics = eval_metrics(
        y_true, y_pred,
        average="macro",
        labels=labels,
        return_report=True,
        return_confusion=True,
    )
    return metrics
    