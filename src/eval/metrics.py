"""
Evaluation utilities — precision, recall, F1, false positive rate,
confusion matrix, and a pretty-print summary.
"""

import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, f1_score
)
from typing import List

LABEL_NAMES = ["no_bug", "null_dereference", "off_by_one", "resource_leak", "logic_error"]


def evaluate(y_true: List[int], y_pred: List[int]) -> dict:
    """Compute full evaluation metrics."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(5))
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    # False positive rate per class
    fp_rates = []
    for i in range(5):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fp_rates.append(fpr)

    results = {
        "macro_f1": float(macro_f1),
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "false_positive_rate": float(fp_rates[i]),
            }
            for i in range(5)
        },
        "confusion_matrix": cm.tolist(),
    }
    return results


def print_report(y_true: List[int], y_pred: List[int]):
    """Pretty-print classification report."""
    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))
    results = evaluate(y_true, y_pred)
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print("\nFalse Positive Rates:")
    for cls, m in results["per_class"].items():
        print(f"  {cls:20s}: {m['false_positive_rate']:.4f}")