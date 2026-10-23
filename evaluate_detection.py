"""Detection validation on the FSOCO bounding-box set: AP over IoU 0.50-0.95, confusion matrix,
recall by distance, all inside the configured range gate.

    python evaluate_detection.py --model path/to/cone_pose.pt --dataset data/fsoco_bounding_boxes_train
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from cone_validation import Config
from cone_validation import plots
from cone_validation.distance import estimate_distance, in_gate
from cone_validation.fsoco import load_cones, samples
from cone_validation.metrics import IOU_THRESHOLDS, average_precision, confusion_matrix, match
from cone_validation.model import check_model, predict

DISTANCE_BINS = [1, 5, 10, 15]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, type=Path, help="FSOCO bounding-box root (<team>/img, <team>/ann)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25, help="score threshold for the confusion matrix and recall")
    p.add_argument("--exclude", type=Path, help="image names to leave out, one per line (e.g. your training images)")
    p.add_argument("--limit", type=int, help="evaluate only the first N images, for a quick check")
    p.add_argument("--out", type=Path, default=Path("results/detection"))
    args = p.parse_args()

    cfg = Config.load(args.config)
    model = YOLO(args.model)
    C = len(cfg.names)
    exclude = args.exclude.read_text().split() if args.exclude else ()
    pairs = list(samples(args.dataset, exclude))[: args.limit]
    if not pairs:
        raise SystemExit(f"no images with annotations under {args.dataset}")
    check_model(model, cfg)

    flags = {c: [] for c in range(C)}           # per class: (tp, ignored, scores)
    n_gt = np.zeros(C, int)
    cm = np.zeros((C + 1, C + 1), int)
    dist_rows = []                              # (distance, detected, right class)

    for image, annotation in tqdm(pairs, desc="images"):
        height, cones = load_cones(annotation, cfg)
        gt_cls = np.array([c.cls for c in cones], int)
        gt_box = np.array([c.box for c in cones], float).reshape(-1, 4)
        unreliable = np.array([bool(c.tags & cfg.unreliable_tags) for c in cones], bool)
        gated = in_gate(gt_box, gt_cls, height, cfg)
        gt_ignore = (gt_cls < 0) | (cfg.gate_enabled & (~gated | unreliable))

        pred = predict(model, image, conf=0.001, imgsz=args.imgsz)
        pred_outside = ~in_gate(pred.boxes, pred.cls, height, cfg)

        for c in range(C):
            pk, gk = pred.cls == c, (gt_cls == c) | (gt_cls < 0)     # unknown cones: ignore regions for every class
            tp, ign, _ = match(pred.boxes[pk], pred.scores[pk], gt_box[gk], gt_ignore[gk], pred_outside[pk])
            flags[c].append((tp, ign, pred.scores[pk]))
            n_gt[c] += int((gk & ~gt_ignore).sum())

        sure = pred.scores >= args.conf
        img_cm, detected_as = confusion_matrix(pred.cls[sure], pred.boxes[sure], pred.scores[sure], gt_cls, gt_box,
                                               gt_ignore, pred_outside[sure], C)
        cm += img_cm
        if len(cones):
            heights = np.array([cfg.heights_m[c] for c in np.clip(gt_cls, 0, None)])
            d = estimate_distance(gt_box[:, 3] - gt_box[:, 1], heights, height, cfg.vertical_fov_deg)
            for g in np.flatnonzero(~gt_ignore):
                dist_rows.append((d[g], detected_as[g] >= 0, detected_as[g] == gt_cls[g]))

    ap = np.full((C, len(IOU_THRESHOLDS)), np.nan)
    for c in range(C):
        if flags[c]:
            tp = np.vstack([f[0] for f in flags[c]])
            ign = np.vstack([f[1] for f in flags[c]])
            ap[c] = average_precision(tp, ign, np.concatenate([f[2] for f in flags[c]]), n_gt[c])

    rows = np.array(dist_rows, float).reshape(-1, 3)
    bins = DISTANCE_BINS
    which = np.clip(np.digitize(rows[:, 0], bins) - 1, -1, len(bins) - 2)   # the gate's upper edge counts as inside
    which[rows[:, 0] < bins[0]] = -1
    counts = [int((which == b).sum()) for b in range(len(bins) - 1)]
    detected = [float(rows[which == b, 1].mean()) if counts[b] else 0.0 for b in range(len(bins) - 1)]
    correct = [float(rows[which == b, 2].mean()) if counts[b] else 0.0 for b in range(len(bins) - 1)]

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "model": str(args.model),
        "images": len(pairs),
        "imgsz": args.imgsz,
        "range_gate_m": [cfg.gate_min_m, cfg.gate_max_m] if cfg.gate_enabled else None,
        "ground_truth_per_class": dict(zip(cfg.names, n_gt.tolist())),
        "mAP50": float(np.nanmean(ap[:, 0])),
        "mAP50_95": float(np.nanmean(ap)),
        "AP50_per_class": {n: None if np.isnan(ap[i, 0]) else float(ap[i, 0]) for i, n in enumerate(cfg.names)},
        "recall_at_conf": {n: float(cm[i, i] / max(cm[:, i].sum(), 1)) for i, n in enumerate(cfg.names)},
        "recall_by_distance": {f"{bins[b]}-{bins[b + 1]} m": {"n": counts[b], "detected": detected[b],
                                                             "right_colour": correct[b]} for b in range(len(bins) - 1)},
        "confusion_matrix": cm.tolist(),
    }
    (args.out / "detection.json").write_text(json.dumps(report, indent=2))
    plots.confusion(cm, cfg.names, args.out / "confusion_matrix.png")
    plots.ap_vs_iou(ap, IOU_THRESHOLDS, cfg.names, args.out / "ap_vs_iou.png")
    plots.recall_by_distance(bins, detected, correct, counts, args.out / "recall_by_distance.png")
    print(f"mAP50 {report['mAP50']:.3f}  mAP50-95 {report['mAP50_95']:.3f}  ->  {args.out}")


if __name__ == "__main__":
    main()
