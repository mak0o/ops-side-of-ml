"""分類性能の指標。

学習時の評価とホールドアウト評価で同じ定義を使うため、ここに集める。
定義が2か所に分かれると、いずれ片方だけ直されて比較が壊れる。
"""

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
    }