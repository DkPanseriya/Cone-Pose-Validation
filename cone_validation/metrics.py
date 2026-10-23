"""Detection metrics with ignore regions: COCO-style greedy matching and 101-point AP."""
import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)


def box_iou(a, b):
    """IoU matrix between xyxy boxes a (N, 4) and b (M, 4)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(pred_boxes, scores, gt_boxes, gt_ignore, pred_outside, thresholds=IOU_THRESHOLDS):
    """Match one image's predictions of one class to its ground truth.

    Highest score first, each prediction takes the unclaimed, non-ignored ground truth with the
    highest IoU. A prediction that finds none is ignored (neither TP nor FP) when it overlaps an
    ignored ground truth, or when it lies outside the range gate itself; otherwise it is a FP.

    Returns tp, ignored: bool arrays (P, T), and gt_matched (G, T).
    """
    P, G, T = len(pred_boxes), len(gt_boxes), len(thresholds)
    tp, ignored, gt_matched = np.zeros((P, T), bool), np.zeros((P, T), bool), np.zeros((G, T), bool)
    if P == 0:
        return tp, ignored, gt_matched
    gt_ignore = np.asarray(gt_ignore, bool)
    iou = box_iou(pred_boxes, gt_boxes) if G else np.zeros((P, 0))
    order = np.argsort(-np.asarray(scores), kind="stable")
    for t, thr in enumerate(thresholds):
        for p in order:
            free = np.where(~gt_matched[:, t] & ~gt_ignore, iou[p], -1.0)
            if G and free.max() >= thr:
                g = int(np.argmax(free))
                tp[p, t] = gt_matched[g, t] = True
            elif (G and np.any(iou[p][gt_ignore] >= thr)) or pred_outside[p]:
                ignored[p, t] = True
    return tp, ignored, gt_matched


def average_precision(tp, ignored, scores, n_gt):
    """AP per threshold (T,), 101-point interpolation with the precision envelope. NaN without ground truth."""
    T = tp.shape[1]
    if n_gt == 0:
        return np.full(T, np.nan)
    order = np.argsort(-np.asarray(scores), kind="stable")
    grid = np.linspace(0, 1, 101)
    ap = np.zeros(T)
    for t in range(T):
        keep = order[~ignored[order, t]]
        hits = tp[keep, t]
        tp_cum, fp_cum = np.cumsum(hits), np.cumsum(~hits)
        recall = tp_cum / n_gt
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        idx = np.searchsorted(recall, grid, side="left")
        ap[t] = np.mean([envelope[i] if i < len(envelope) else 0.0 for i in idx])
    return ap


def confusion_matrix(pred_cls, pred_boxes, pred_scores, gt_cls, gt_boxes, gt_ignore, pred_outside, n_classes, iou=0.5):
    """(C+1, C+1) counts for one image, rows = predicted, columns = ground truth, last index = background.

    Class-agnostic one-to-one matching by score, so a cone predicted with the wrong colour shows up
    off the diagonal instead of as a miss plus a false positive.
    Also returns, per ground truth, the class it was detected as (-1 when missed).
    """
    cm = np.zeros((n_classes + 1, n_classes + 1), int)
    bg = n_classes
    detected_as = np.full(len(gt_boxes), -1)
    ious = box_iou(pred_boxes, gt_boxes) if len(gt_boxes) else np.zeros((len(pred_boxes), 0))
    gt_ignore = np.asarray(gt_ignore, bool)
    for p in np.argsort(-np.asarray(pred_scores), kind="stable"):
        free = np.where((detected_as < 0) & ~gt_ignore, ious[p], -1.0)
        if len(gt_boxes) and free.max() >= iou:
            g = int(np.argmax(free))
            detected_as[g] = pred_cls[p]
            cm[pred_cls[p], gt_cls[g]] += 1
        elif (len(gt_boxes) and np.any(ious[p][gt_ignore] >= iou)) or pred_outside[p]:
            continue                                   # on an ignored cone, or outside the gate: not counted
        else:
            cm[pred_cls[p], bg] += 1
    for g in np.flatnonzero((detected_as < 0) & ~gt_ignore):
        cm[bg, gt_cls[g]] += 1
    return cm, detected_as
