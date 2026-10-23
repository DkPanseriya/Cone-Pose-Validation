"""Keypoint accuracy against cone outlines taken from segmentation masks.

FSOCO has no keypoint labels, but it has pixel masks. Every keypoint the model predicts lies on the
cone's outline, so its distance to the mask outline is a labelling-free measure of how well it sits.
"""
import cv2
import numpy as np

from .metrics import box_iou


def match_to_masks(pred_cls, pred_boxes, pred_scores, cones, iou=0.5):
    """One-to-one, same class, highest score first. Returns [(pred index, cone index)]."""
    usable = [i for i, c in enumerate(cones) if c.cls >= 0]
    if not usable or len(pred_boxes) == 0:
        return []
    ious = box_iou(pred_boxes, [cones[i].box for i in usable])
    taken, pairs = set(), []
    for p in np.argsort(-np.asarray(pred_scores), kind="stable"):
        best, best_iou = None, iou
        for j, ci in enumerate(usable):
            if ci not in taken and cones[ci].cls == pred_cls[p] and ious[p, j] >= best_iou:
                best, best_iou = ci, ious[p, j]
        if best is not None:
            taken.add(best)
            pairs.append((int(p), best))
    return pairs


def outline_distance(point, contours):
    """Pixel distance from a point to the nearest outline of the mask (0 on the outline)."""
    pt = (float(point[0]), float(point[1]))
    return min(abs(cv2.pointPolygonTest(c.astype(np.float32).reshape(-1, 1, 2), pt, True)) for c in contours)


def order_ok(kpts):
    """Plausibility of the keypoint layout, which outline distance alone cannot see: every left point
    lies left of its right partner, and each side climbs from bottom to top. Catches swapped or
    mirrored keypoints that still sit on the outline."""
    k = np.asarray(kpts, float)
    n = len(k)
    left_of_right = all(k[i, 0] < k[n - 1 - i, 0] for i in range(n // 2))
    left_climbs = all(k[i, 1] > k[i + 1, 1] for i in range(n // 2 - 1))
    right_climbs = all(k[n - 1 - i, 1] > k[n - 2 - i, 1] for i in range(n // 2 - 1))
    return bool(left_of_right and left_climbs and right_climbs)


def keypoint_errors(kpts, cone):
    """Per-keypoint outline distance in pixels and as a fraction of the cone's height."""
    px = np.array([outline_distance(p, cone.contours) for p in kpts])
    return px, px / max(cone.box[3] - cone.box[1], 1.0)
