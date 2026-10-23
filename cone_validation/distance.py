"""Monocular distance from the known cone height, for the range gate."""
import numpy as np


def focal_length_px(image_height, vertical_fov_deg):
    return (image_height / 2) / np.tan(np.radians(vertical_fov_deg) / 2)


def estimate_distance(box_height_px, cone_height_m, image_height, vertical_fov_deg):
    """Pinhole estimate: distance = real height * focal length / height in pixels."""
    box_height_px = np.asarray(box_height_px, float)
    with np.errstate(divide="ignore"):
        return np.where(box_height_px > 0,
                        cone_height_m * focal_length_px(image_height, vertical_fov_deg) / box_height_px, np.inf)


def in_gate(boxes, classes, image_height, config):
    """True for boxes whose estimated distance lies inside the configured range gate."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    if not config.gate_enabled or len(boxes) == 0:
        return np.ones(len(boxes), bool)
    heights = np.array([config.heights_m[c] if c >= 0 else config.heights_m[-1] for c in classes])
    d = estimate_distance(boxes[:, 3] - boxes[:, 1], heights, image_height, config.vertical_fov_deg)
    return (d >= config.gate_min_m) & (d <= config.gate_max_m)
