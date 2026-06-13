from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


def evaluate_model(y_true, y_pred, y_prob=None):
    """
    Evaluate a classification model.

    Returns
    -------
    acc           : float   — overall accuracy
    precision_w   : float   — weighted precision
    recall_w      : float   — weighted recall
    f1_w          : float   — weighted F1 (kept for backward-compat / best-model gate)
    f1_macro      : float   — macro F1 (treats both classes equally; used for NIDS fairness)
    recall_attack : float   — recall on the attack class (class 1); minimises missed attacks
    roc_auc       : float|None — ROC-AUC when probabilities are supplied, else None
    cm            : ndarray — 2x2 confusion matrix [[TN,FP],[FN,TP]]
    report        : str     — full per-class precision / recall / F1 report
    """

    acc         = accuracy_score(y_true, y_pred)
    precision_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_w    = recall_score(y_true, y_pred,    average="weighted", zero_division=0)
    f1_w        = f1_score(y_true, y_pred,        average="weighted", zero_division=0)

    # Fix Step 9: macro F1 gives equal weight to both classes
    f1_macro      = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Fix Step 9: recall on attack class (class 1) — false negatives are missed attacks
    recall_attack = recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)

    # Fix Step 9: ROC-AUC when probabilities are available
    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            pass

    # Fix Step 9: confusion matrix and per-class report
    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred,
                                   target_names=["Normal", "Attack"],
                                   zero_division=0)

    return acc, precision_w, recall_w, f1_w, f1_macro, recall_attack, roc_auc, cm, report