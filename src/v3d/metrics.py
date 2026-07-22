"""Evaluation metrics for Phase 1 reproduction.

Detection (single "ball" class): average precision and recall at a fixed IoU
threshold, matching predictions to at most one ground-truth box per frame,
greedily in descending confidence order. AP is the area under the
precision-recall curve using all-points interpolation (COCO / VOC2010+ style).

3D localization: Euclidean error in meters between an estimated ball position
and the triangulated ground truth, reported as mean / median / percentiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two [x, y, w, h] boxes (top-left corner + size)."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class DetectionResult:
    ap: float          # average precision at the IoU threshold
    recall: float      # TP / n_gt at the best-F1 operating point
    precision: float   # precision at that same operating point
    best_f1: float
    best_conf: float   # confidence threshold at best F1
    n_gt: int
    n_pred: int
    iou_thr: float


def evaluate_detections(
    per_frame: list[dict],
    iou_thr: float = 0.5,
) -> DetectionResult:
    """Compute AP and best-F1 recall/precision for single-class ball detection.

    per_frame: list of {"gt": [x,y,w,h] or None,
                        "preds": [([x,y,w,h], conf), ...]}.
    One GT ball per frame at most (the SNv3D annotation).
    """
    entries = []  # (conf, is_tp)
    n_gt = 0
    n_pred = 0
    for fr in per_frame:
        gt = fr.get("gt")
        preds = sorted(fr.get("preds", []), key=lambda p: -p[1])
        n_pred += len(preds)
        if gt is not None:
            n_gt += 1
        matched = False
        for box, conf in preds:
            is_tp = False
            if gt is not None and not matched and iou_xywh(box, gt) >= iou_thr:
                is_tp = True
                matched = True
            entries.append((conf, is_tp))

    if not entries:
        return DetectionResult(0.0, 0.0, 0.0, 0.0, 0.0, n_gt, n_pred, iou_thr)

    entries.sort(key=lambda e: -e[0])
    tp = np.array([1 if e[1] else 0 for e in entries])
    fp = 1 - tp
    confs = np.array([e[0] for e in entries])
    ctp = np.cumsum(tp)
    cfp = np.cumsum(fp)
    recalls = ctp / max(n_gt, 1)
    precisions = ctp / np.maximum(ctp + cfp, 1e-9)

    ap = _ap_all_points(recalls, precisions)

    f1 = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-9)
    bi = int(np.argmax(f1))
    return DetectionResult(
        ap=float(ap),
        recall=float(recalls[bi]),
        precision=float(precisions[bi]),
        best_f1=float(f1[bi]),
        best_conf=float(confs[bi]),
        n_gt=n_gt,
        n_pred=n_pred,
        iou_thr=iou_thr,
    )


def _ap_all_points(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Area under the PR curve with all-points interpolation."""
    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[0.0], precisions, [0.0]])
    # Make precision monotonically decreasing from the right.
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def localization_error_stats(errors_m: np.ndarray, p2m_thresh: float = 2.0) -> dict[str, float]:
    """Summarize 3D localization errors (meters).

    Includes P2m (the paper's `P2m` metric): the fraction of estimates within
    `p2m_thresh` meters of the ground-truth ball position.
    """
    e = np.asarray(errors_m, dtype=float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"n": 0}
    return {
        "n": int(e.size),
        "mean_m": float(np.mean(e)),
        "median_m": float(np.median(e)),
        "p90_m": float(np.percentile(e, 90)),
        "rmse_m": float(np.sqrt(np.mean(e**2))),
        "max_m": float(np.max(e)),
        "p2m": float(np.mean(e <= p2m_thresh)),
    }
