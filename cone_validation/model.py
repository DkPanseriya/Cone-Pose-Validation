"""Run an Ultralytics YOLO model and return plain arrays."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Predictions:
    cls: np.ndarray        # (N,)
    boxes: np.ndarray      # (N, 4) xyxy pixels
    scores: np.ndarray     # (N,)
    keypoints: np.ndarray  # (N, K, 2) pixels, or None for a detect-only model


def check_model(model, config, need_keypoints=False):
    """Stop early when the model does not fit config.yaml, instead of scoring the wrong classes."""
    names = [model.names[i] for i in sorted(model.names)]
    if len(names) != len(config.names):
        raise SystemExit(f"model has {len(names)} classes {names}, config.yaml has {len(config.names)}")
    print("class mapping (model -> config):", ", ".join(f"{a} -> {b}" for a, b in zip(names, config.names)))
    kpt_shape = getattr(model.model, "yaml", {}).get("kpt_shape")
    if need_keypoints and (kpt_shape is None or kpt_shape[0] != len(config.keypoints)):
        raise SystemExit(f"model keypoints {kpt_shape} do not match the {len(config.keypoints)} in config.yaml")


def predict(model, image, conf, imgsz):
    r = model.predict(str(image), conf=conf, imgsz=imgsz, iou=0.7, max_det=300, verbose=False)[0]
    kpts = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
    return Predictions(r.boxes.cls.cpu().numpy().astype(int), r.boxes.xyxy.cpu().numpy(),
                       r.boxes.conf.cpu().numpy(), kpts)
