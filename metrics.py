"""
utils/metrics.py
Evaluation metrics for semantic change detection.
  - mIoU (mean Intersection over Union)
  - OA   (Overall Accuracy)
  - F1   (per-class and macro)
  - SeK  (Separated Kappa — Yang et al., IEEE TGRS 2023)
"""

import torch
import numpy as np
from typing import Dict, Optional


class SCDMetrics:
    """
    Online accumulator for SCD evaluation metrics.

    Maintains a confusion matrix Q of shape (C, C) where
    Q[i, j] = number of pixels with true class i predicted as class j.

    The Separated Kappa (SeK) coefficient separates the non-change class
    (index 0 by convention) from semantic change classes to produce a
    bias-corrected evaluation score under severe label imbalance.

    Args:
        num_classes  : total number of semantic classes (including background)
        ignore_index : label value to ignore (e.g. 255 for void pixels)
    """

    def __init__(self, num_classes: int = 9, ignore_index: int = 255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Clear the accumulated confusion matrix."""
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Accumulate predictions into the confusion matrix.

        Args:
            pred   : (B, H, W) predicted class indices (long tensor)
            target : (B, H, W) ground-truth class indices (long tensor)
        """
        pred   = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()

        valid  = target != self.ignore_index
        pred   = pred[valid]
        target = target[valid]

        # Clip to valid range
        pred   = np.clip(pred,   0, self.num_classes - 1)
        target = np.clip(target, 0, self.num_classes - 1)

        np.add.at(self.confusion, (target, pred), 1)

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from the accumulated confusion matrix.

        Returns:
            dict with keys: 'mIoU', 'OA', 'macro_F1', 'SeK',
                            'per_class_IoU', 'per_class_F1'
        """
        Q = self.confusion.astype(np.float64)
        C = self.num_classes

        # ── OA ──────────────────────────────────────────────────────────
        oa = Q.diagonal().sum() / (Q.sum() + 1e-8)

        # ── Per-class IoU ────────────────────────────────────────────────
        tp  = Q.diagonal()
        fp  = Q.sum(axis=0) - tp
        fn  = Q.sum(axis=1) - tp
        iou = tp / (tp + fp + fn + 1e-8)

        # ── Per-class F1 ─────────────────────────────────────────────────
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)

        # ── mIoU (macro over all classes, skip void) ─────────────────────
        miou = iou.mean()

        # ── Separated Kappa (SeK) — Yang et al. 2023 ────────────────────
        #
        # IoU_1  measures non-change class accuracy (class index 0)
        # IoU_2  measures semantic change class accuracy (classes 1..C-1)
        #
        # IoU_1 = Q[0,0] / (Σ_i Q[i,0] + Σ_j Q[0,j] − Q[0,0])
        # IoU_2 = Σ_{i≥1,j≥1} Q[i,j] / (Σ_{i,j} Q[i,j] − Q[0,0])
        # SeK   = IoU_2 · (1 − IoU_1) / (1 − IoU_1 · IoU_2)

        q00   = Q[0, 0]
        denom1 = Q[:, 0].sum() + Q[0, :].sum() - q00
        iou1   = q00 / (denom1 + 1e-8)

        change_sum  = Q[1:, 1:].sum()
        total_sum   = Q.sum() - q00
        iou2        = change_sum / (total_sum + 1e-8)

        sek_denom = 1 - iou1 * iou2
        if sek_denom < 1e-8:
            sek = 0.0
        else:
            sek = iou2 * (1 - iou1) / sek_denom

        return {
            'mIoU':         float(miou * 100),
            'OA':           float(oa   * 100),
            'macro_F1':     float(f1.mean() * 100),
            'SeK':          float(sek  * 100),
            'IoU_1':        float(iou1 * 100),   # non-change IoU
            'IoU_2':        float(iou2 * 100),   # semantic change IoU
            'per_class_IoU': (iou * 100).tolist(),
            'per_class_F1':  (f1  * 100).tolist(),
        }

    def summary(self,
                class_names: Optional[list] = None,
                decimals: int = 2) -> str:
        """
        Return a human-readable summary string.

        Args:
            class_names : list of class name strings (optional)
            decimals    : number of decimal places
        """
        m = self.compute()
        lines = [
            '─' * 52,
            f"  mIoU      : {m['mIoU']:.{decimals}f} %",
            f"  OA        : {m['OA']:.{decimals}f} %",
            f"  Macro F1  : {m['macro_F1']:.{decimals}f} %",
            f"  SeK       : {m['SeK']:.{decimals}f} pp",
            f"    IoU_1   : {m['IoU_1']:.{decimals}f} %  (non-change)",
            f"    IoU_2   : {m['IoU_2']:.{decimals}f} %  (semantic change)",
            '─' * 52,
            '  Per-class IoU:',
        ]
        for i, v in enumerate(m['per_class_IoU']):
            name = class_names[i] if class_names else f'Class {i}'
            lines.append(f"    [{i:2d}] {name:<22s} {v:.{decimals}f} %")
        lines.append('─' * 52)
        return '\n'.join(lines)
