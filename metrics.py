from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass
class GCDMetrics:
    all_acc: float
    old_acc: float
    new_acc: float
    nmi: float
    ari: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def hungarian_mapping(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, int]:
    true_vals = np.unique(y_true)
    pred_vals = np.unique(y_pred)
    tmap = {v: i for i, v in enumerate(true_vals)}
    pmap = {v: i for i, v in enumerate(pred_vals)}
    mat = np.zeros((len(pred_vals), len(true_vals)), dtype=np.int64)
    for p, t in zip(y_pred, y_true):
        mat[pmap[p], tmap[t]] += 1
    r, c = linear_sum_assignment(mat.max() - mat)
    return {int(pred_vals[ri]): int(true_vals[ci]) for ri, ci in zip(r, c)}


def evaluate_gcd(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    old_classes: np.ndarray,
) -> Tuple[GCDMetrics, Dict[int, int]]:
    """V2 protocol: compute Hungarian assignment ONCE on all evaluation samples."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mapping = hungarian_mapping(y_true, y_pred)
    mapped = np.array([mapping.get(int(p), -10**9) for p in y_pred])
    correct = mapped == y_true
    old_mask = np.isin(y_true, old_classes)
    new_mask = ~old_mask
    old_acc = float(correct[old_mask].mean()) if old_mask.any() else float("nan")
    new_acc = float(correct[new_mask].mean()) if new_mask.any() else float("nan")
    m = GCDMetrics(
        all_acc=float(correct.mean()),
        old_acc=old_acc,
        new_acc=new_acc,
        nmi=float(normalized_mutual_info_score(y_true, y_pred)),
        ari=float(adjusted_rand_score(y_true, y_pred)),
    )
    return m, mapping
