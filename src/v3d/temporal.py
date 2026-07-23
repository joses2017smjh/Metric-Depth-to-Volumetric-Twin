"""Track 1: temporal aggregation of single-frame ball detections.

Phase 1 measured a recall ceiling of ~0.65 for the released single-frame
detector: in a third of frames it never proposes the ball at all, typically
when motion blur smears a fast ball into a streak. Neighbouring video frames
usually *do* show it, because blur direction and background clutter change from
frame to frame.

This module exploits that without retraining. Given detections on a window of
consecutive frames, it fits a constant-velocity image-space track by RANSAC,
then reports the ball at the centre frame either by confirming the centre
frame's own detection or, when the detector missed it there, by interpolating
the track. A short window (a few hundred ms) is well described by constant
velocity even for a fast ball, so the model stays simple and robust.

Detections are `(box_xywh, confidence)` grouped by integer frame offset k,
with k = 0 the annotated keyframe.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


def box_center(box) -> np.ndarray:
    b = np.asarray(box, float)
    return np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])


@dataclass
class Track:
    p0: np.ndarray        # image position at k = 0 (pixels)
    velocity: np.ndarray  # pixels per frame
    inliers: list         # (k, box, conf) supporting the model
    size: np.ndarray      # median (w, h) of inlier boxes
    n_frames: int         # frames in the window

    @property
    def support(self) -> int:
        return len(self.inliers)

    def position_at(self, k: float) -> np.ndarray:
        return self.p0 + self.velocity * k

    def box_at(self, k: float) -> np.ndarray:
        c = self.position_at(k)
        return np.array([c[0] - self.size[0] / 2, c[1] - self.size[1] / 2,
                         self.size[0], self.size[1]])


def fit_track(
    dets_by_k: dict[int, list],
    tol_px: float = 12.0,
    min_support: int = 3,
    max_speed_px: float = 250.0,
    min_conf: float = 0.10,
    min_speed_px: float = 0.0,
) -> Track | None:
    """RANSAC a constant-velocity image track through per-frame detections.

    Every pair of detections from two different frames proposes a model
    (position, velocity); `tol_px` is the per-frame association radius and
    `max_speed_px` rejects pairings implying implausible ball motion.

    Models are ranked by the **summed confidence** of their inliers, not by
    inlier count. That distinction matters: run at a low confidence floor the
    detector emits many spurious boxes on static background clutter, and a
    static distractor sits at the same pixel in every frame, forming a perfect
    zero-velocity "track" that wins any count-based vote. Confidence weighting
    lets a shorter track of genuine, confident ball detections beat a long
    track of weak clutter. `min_conf` additionally keeps the candidate pool
    from being swamped, and `min_speed_px` can require actual motion (a truly
    static ball is already handled well by the single-frame detector).
    """
    pool = [
        (k, box, conf)
        for k, ds in dets_by_k.items()
        for box, conf in ds
        if conf >= min_conf
    ]
    if len(pool) < min_support:
        return None

    best: Track | None = None
    best_score = -np.inf
    for (k1, b1, _), (k2, b2, _) in combinations(pool, 2):
        if k1 == k2:
            continue
        c1, c2 = box_center(b1), box_center(b2)
        v = (c2 - c1) / (k2 - k1)
        speed = float(np.linalg.norm(v))
        if speed > max_speed_px or speed < min_speed_px:
            continue
        p0 = c1 - v * k1  # position at k = 0

        # One inlier per frame: the detection closest to the prediction.
        inliers = []
        for k, ds in dets_by_k.items():
            pred = p0 + v * k
            cand = [
                (np.linalg.norm(box_center(b) - pred), b, c)
                for b, c in ds
                if c >= min_conf
            ]
            if not cand:
                continue
            dist, b, c = min(cand, key=lambda t: t[0])
            if dist <= tol_px:
                inliers.append((k, b, c))
        if len(inliers) < min_support:
            continue
        sizes = np.array([[b[2], b[3]] for _, b, _ in inliers], float)
        # Three signals, multiplied:
        #   confidence  - genuine ball detections score higher than clutter
        #   size stability - a real ball keeps its apparent size across a short
        #                    window, whereas clutter associations jump around
        #   fit residual  - how tightly the detections follow the linear model
        conf_sum = float(sum(c for _, _, c in inliers))
        mean_size = float(sizes.mean()) or 1.0
        size_cv = float(sizes.std() / mean_size)
        resid = float(np.mean([
            np.linalg.norm(box_center(b) - (p0 + v * k)) for k, b, _ in inliers
        ]))
        score = conf_sum / (1.0 + size_cv) / (1.0 + resid / max(tol_px, 1e-6))
        if score > best_score:
            best_score = score
            best = Track(p0=p0, velocity=v, inliers=inliers,
                         size=np.median(sizes, axis=0), n_frames=len(dets_by_k))

    if best is None:
        return None
    # Refit the winning model on its inliers (least squares) for stability.
    ks = np.array([k for k, _, _ in best.inliers], float)
    cs = np.array([box_center(b) for _, b, _ in best.inliers])
    A = np.vstack([np.ones_like(ks), ks]).T
    coef, *_ = np.linalg.lstsq(A, cs, rcond=None)
    best.p0, best.velocity = coef[0], coef[1]
    return best


def aggregate_window(
    dets_by_k: dict[int, list],
    tol_px: float = 12.0,
    min_support: int = 3,
    recover_conf_scale: float = 1.0,
    min_conf: float = 0.10,
    min_speed_px: float = 0.0,
    refine: bool = True,
) -> list:
    """Detections for the centre frame, augmented by temporal evidence.

    Returns a list of (box_xywh, confidence) for k = 0:

    * the centre frame's own detections are kept; one consistent with the track
      is *boosted*, since independent frames agreeing on a moving ball is strong
      evidence;
    * if the centre frame has no detection on the track, the track's
      interpolated box is emitted, with a confidence derived from how many
      frames support it and how confident those detections were. This is the
      case that recovers a ball the single-frame detector missed entirely.
    """
    own = list(dets_by_k.get(0, []))
    track = fit_track(dets_by_k, tol_px=tol_px, min_support=min_support,
                      min_conf=min_conf, min_speed_px=min_speed_px)
    if track is None:
        return own

    support_frac = track.support / max(track.n_frames, 1)
    mean_conf = float(np.mean([c for _, _, c in track.inliers]))

    pred_c = track.position_at(0)
    matched_idx = None
    for i, (b, _) in enumerate(own):
        if np.linalg.norm(box_center(b) - pred_c) <= tol_px:
            matched_idx = i
            break

    if matched_idx is not None:
        out = []
        boosted = float(min(1.0, own[matched_idx][1] + (1.0 - own[matched_idx][1]) * support_frac))
        for i, (b, c) in enumerate(own):
            if i == matched_idx:
                # Replace the single-frame box with the track's estimate when
                # asked. On ~15 px balls IoU@0.5 is dominated by a few pixels of
                # position/size error, and the track pools evidence from the
                # whole window, so its box is usually better localized than any
                # individual frame's.
                box = track.box_at(0) if refine else b
                out.append((box, boosted))
            else:
                out.append((b, c))
        return out

    # Detector missed the ball on the centre frame — recover it from the track.
    rec_conf = float(min(1.0, mean_conf * support_frac * recover_conf_scale))
    return own + [(track.box_at(0), rec_conf)]
