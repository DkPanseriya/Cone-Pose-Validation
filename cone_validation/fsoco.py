"""Read the FSOCO dataset (Supervisely format: <team>/img/<image>, <team>/ann/<image>.json)."""
import base64
import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class Cone:
    cls: int                  # model class id, -1 for ignored classes
    box: np.ndarray           # (4,) xyxy pixels
    tags: set = field(default_factory=set)
    contours: list = None     # outline polygons in image pixels (segmentation set only)


def samples(root, exclude=()):
    """Yield (image_path, annotation_path) pairs, skipping image names listed in `exclude`."""
    exclude = set(exclude)
    for image in sorted(Path(root).rglob("*")):
        if image.suffix.lower() not in IMAGE_SUFFIXES or image.parent.name != "img" or image.name in exclude:
            continue
        annotation = image.parent.parent / "ann" / (image.name + ".json")
        if annotation.exists():
            yield image, annotation


def decode_bitmap(data):
    """Supervisely bitmap: base64 -> zlib -> PNG whose alpha channel is the mask."""
    png = cv2.imdecode(np.frombuffer(zlib.decompress(base64.b64decode(data)), np.uint8), cv2.IMREAD_UNCHANGED)
    return (png[..., 3] if png.ndim == 3 and png.shape[2] == 4 else png.reshape(png.shape[0], png.shape[1], -1)[..., 0]) > 0


def outline(mask, origin, upsample=4):
    """Mask outline in continuous image coordinates (pixel i spans [i, i+1]), the convention the
    model's keypoints use. Tracing an upsampled mask puts the outline within 1/(2*upsample) px of
    the true pixel edge on every side, instead of half a pixel inside on the right and bottom only."""
    big = cv2.resize(mask.astype(np.uint8), None, fx=upsample, fy=upsample, interpolation=cv2.INTER_NEAREST)
    found, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [(c.reshape(-1, 2) + 0.5) / upsample + np.asarray(origin, float) for c in found]


def load_cones(annotation, config):
    """Ground-truth cones of one image, with boxes, tags and (for bitmaps) outlines."""
    data = json.loads(Path(annotation).read_text())
    cones = []
    for obj in data.get("objects", []):
        title = obj["classTitle"]
        if title not in config.fsoco_to_class and title not in config.ignore_classes:
            continue
        cls = config.fsoco_to_class.get(title, -1)
        tags = {t["name"] for t in obj.get("tags", [])}
        if obj["geometryType"] == "rectangle":
            (x1, y1), (x2, y2) = obj["points"]["exterior"]
            box = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], float)
            cones.append(Cone(cls, box, tags))
        elif obj["geometryType"] == "bitmap":
            mask = decode_bitmap(obj["bitmap"]["data"])
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            ox, oy = obj["bitmap"]["origin"]
            box = np.array([xs.min() + ox, ys.min() + oy, xs.max() + 1 + ox, ys.max() + 1 + oy], float)
            cones.append(Cone(cls, box, tags, outline(mask, (ox, oy))))
    return data["size"]["height"], cones
