from typing import Any, Dict, Optional, Sequence
import json
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
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
        A dictionary with 'malignant' and 'multiclass' keys, each containing metrics.
    """
    labels = labels or sorted(set(y_true + y_pred))
    
    # ==================== BINARY CLASSIFICATION (MALIGNANT vs NON-MALIGNANT) ====================
    # Convert to binary for malignant vs non-malignant
    y_true_binary = [1 if label == "malignant" else 0 for label in y_true]
    y_pred_binary = [1 if label == "malignant" else 0 for label in y_pred]
    
    # Binary confusion matrix
    conf_matrix_binary = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    
    # Binary metrics
    acc_malignant = accuracy_score(y_true_binary, y_pred_binary)
    precision_malignant, recall_malignant, f1_malignant, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average="binary", zero_division=0
    )
    
    # Binary sensitivity and specificity
    tn, fp, fn, tp = conf_matrix_binary.ravel()
    sensitivity_malignant = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity_malignant = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # ==================== MULTICLASS CLASSIFICATION ====================
    # Multiclass confusion matrix
    conf_matrix_multiclass = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Multiclass metrics
    acc_multiclass = accuracy_score(y_true, y_pred)
    precision_multiclass, recall_multiclass, f1_multiclass, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, labels=labels, zero_division=0
    )
    
    # Multiclass sensitivity and specificity (macro average)
    sensitivity_multiclass = recall_multiclass  # sensitivity = recall for each class
    specificity_multiclass = []
    for i, label in enumerate(labels):
        # For each class, calculate specificity as TN/(TN+FP) where this class is positive
        tp_i = conf_matrix_multiclass[i, i]
        fp_i = conf_matrix_multiclass[:, i].sum() - tp_i
        tn_i = conf_matrix_multiclass.sum() - conf_matrix_multiclass[i, :].sum() - fp_i
        spec_i = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0
        specificity_multiclass.append(spec_i)
    specificity_multiclass = sum(specificity_multiclass) / len(specificity_multiclass)  # macro average

    # ==================== ASSEMBLE RESULTS ====================
    result = {
        'malignant': {
            'accuracy': acc_malignant,
            'f1': f1_malignant,
            'sensitivity': sensitivity_malignant,
            'specificity': specificity_malignant,
            'precision': precision_malignant,
            'confusion': conf_matrix_binary.tolist() if return_confusion else None,
        },
        'multiclass': {
            'accuracy': acc_multiclass,
            'f1': f1_multiclass,
            'sensitivity': sensitivity_multiclass,
            'specificity': specificity_multiclass,
            'precision': precision_multiclass,
            'confusion': conf_matrix_multiclass.tolist() if return_confusion else None,
        }
    }

    if return_report:
        result['report'] = classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        )

    return result


from tqdm import tqdm
import pandas as pd
from model_skingpt4 import *
import re
def _normalize(s: str, target: str) -> str:
    if "###NLL:" in s:
        s = s.split("###NLL:")[0]
    if target == "y3":
        out = str(s).strip().lower()
        if "malignant" in out:
            return "malignant"
        if "benign" in out:
            return "benign"
        if "other" in out:
            return "other"
        return "unknown"
    if target.startswith('text'):
        out = str(s).strip().lower()
        if "malignant" in out:
            return "malignant"
        if "benign" in out:
            return "benign"
        if ("other" in out) or ("unknown" in out):
            return "other"
        return "unknown"

def _normalize_nll(s: str) -> Dict[str, float]:
    import json
    i = s.rfind("###NLL:")
    if i == -1:
        return {}
    try:
        return json.loads(s[i+len("###NLL:"):].strip())
    except Exception:
        return {}


def compute_auroc_auprc_from_nll(
    y_true: Sequence[str],
    responses: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    score_by = "prob", # "sum_nll" or "avg_nll" or "prob"
) -> Dict[str, Any]:
    labels = list(labels) if labels is not None else sorted(set(y_true))
    y_true_by_label = {lbl: [] for lbl in labels}
    scores_by_label = {lbl: [] for lbl in labels}
    for yt, resp in zip(y_true, responses):
        nll = _normalize_nll(resp)
        if not isinstance(nll, dict):
            continue
        for lbl in labels:
            d = nll.get(lbl)
            if isinstance(d, dict) and (score_by in d):
                y_true_by_label[lbl].append(1 if yt == lbl else 0)
                if score_by == "prob":
                    scores_by_label[lbl].append(float(d[score_by]))
                else:
                    scores_by_label[lbl].append(-float(d[score_by])) # negative the nll to get ll

    per_label_auroc, per_label_auprc = {}, {}
    for lbl in labels:
        ys = y_true_by_label[lbl]
        ss = scores_by_label[lbl]
        if len(ys) >= 2 and len(set(ys)) > 1 and len(ss) == len(ys):
            try:
                per_label_auroc[lbl] = float(roc_auc_score(ys, ss))
            except Exception:
                per_label_auroc[lbl] = None
            try:
                per_label_auprc[lbl] = float(average_precision_score(ys, ss))
            except Exception:
                per_label_auprc[lbl] = None
        else:
            per_label_auroc[lbl] = None
            per_label_auprc[lbl] = None

    malignant = {
        "auroc": per_label_auroc.get("malignant"),
        "auprc": per_label_auprc.get("malignant"),
    }
    macro_vals_roc = [v for v in per_label_auroc.values() if v is not None]
    macro_vals_prc = [v for v in per_label_auprc.values() if v is not None]
    multiclass = {
        "auroc": float(sum(macro_vals_roc) / len(macro_vals_roc)) if macro_vals_roc else None,
        "auprc": float(sum(macro_vals_prc) / len(macro_vals_prc)) if macro_vals_prc else None,
    }
    return {
        "malignant": malignant,
        "multiclass": multiclass,
    }

def eval_ft_skingpt4(chat, dataset, temperature=0.0, remove_system=True,
                     target="y3", 
                     prompt_keys = None,
                     train_mode=True):
    labels = sorted({_normalize(str(dataset[i]['y'][target]), target) for i in range(len(dataset))})
    rows = []
    y_true, y_pred = [], []
    responses = []
    base_question = chat.model.conv_question
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        img = sample['image']
        gt = _normalize(sample['y'][target], target)
        local_q = base_question
        if prompt_keys is not None:
            pre = " ".join(
                (str(sample['y'].get(k, "")).strip() for k in prompt_keys if sample['y'].get(k))
            )
            if pre:
                pre = re.sub(r'\.{2,}', '.', pre)
                local_q = f"{pre} {base_question}"
        resp = chat_with_image(chat, img, local_q, temperature=temperature, remove_system=remove_system, train_mode=train_mode)
        pred = _normalize(resp, target)
        responses.append(resp)
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
    # Attach AUROC/AUPRC computed from NLL footer
    nll_aucs = compute_auroc_auprc_from_nll(y_true, responses, labels)
    if 'malignant' in metrics:
        metrics['malignant']['auroc'] = nll_aucs['malignant']['auroc']
        metrics['malignant']['auprc'] = nll_aucs['malignant']['auprc']
    if 'multiclass' in metrics:
        metrics['multiclass']['auroc'] = nll_aucs['multiclass']['auroc']
        metrics['multiclass']['auprc'] = nll_aucs['multiclass']['auprc']
    return metrics
    