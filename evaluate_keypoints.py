"""Keypoint validation on the FSOCO segmentation set: how far each predicted keypoint lies from the
cone outline in the mask, without a single hand-labelled keypoint.

    python evaluate_keypoints.py --model path/to/cone_pose.pt --dataset data/fsoco_segmentation_train
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from cone_validation import Config
from cone_validation import plots
from cone_validation.distance import in_gate
from cone_validation.fsoco import load_cones, samples
from cone_validation.keypoints import keypoint_errors, match_to_masks, order_ok
from cone_validation.model import check_model, predict


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, type=Path, help="FSOCO segmentation root (<team>/img, <team>/ann)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25, help="score threshold for predictions matched to masks")
    p.add_argument("--exclude", type=Path, help="image names to leave out, one per line")
    p.add_argument("--limit", type=int, help="evaluate only the first N images, for a quick check")
    p.add_argument("--out", type=Path, default=Path("results/keypoints"))
    args = p.parse_args()

    cfg = Config.load(args.config)
    model = YOLO(args.model)
    exclude = args.exclude.read_text().split() if args.exclude else ()
    pairs = list(samples(args.dataset, exclude))[: args.limit]
    if not pairs:
        raise SystemExit(f"no images with annotations under {args.dataset}")
    check_model(model, cfg, need_keypoints=True)

    px = {n: [] for n in cfg.keypoints}
    rel = {n: [] for n in cfg.keypoints}
    masks = matched = layout_ok = 0

    for image, annotation in tqdm(pairs, desc="images"):
        height, cones = load_cones(annotation, cfg)
        keep = [c for c in cones if c.cls >= 0 and not (c.tags & cfg.unreliable_tags)]
        if cfg.gate_enabled and keep:
            gated = in_gate([c.box for c in keep], [c.cls for c in keep], height, cfg)
            keep = [c for c, g in zip(keep, gated) if g]
        masks += len(keep)
        pred = predict(model, image, conf=args.conf, imgsz=args.imgsz)
        if pred.keypoints is None:
            raise SystemExit("this model predicts no keypoints; use a pose model")
        for p_idx, c_idx in match_to_masks(pred.cls, pred.boxes, pred.scores, keep):
            kpts = pred.keypoints[p_idx]
            err_px, err_rel = keypoint_errors(kpts, keep[c_idx])
            for name, a, b in zip(cfg.keypoints, err_px, err_rel):
                px[name].append(float(a))
                rel[name].append(float(b))
            matched += 1
            layout_ok += order_ok(kpts)

    if not matched:
        raise SystemExit("no prediction matched a mask; check the model, the config and the dataset path")

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "model": str(args.model),
        "images": len(pairs),
        "masks_evaluated": masks,
        "masks_matched": matched,
        "plausible_layout": layout_ok / matched,
        "per_keypoint": {n: {"median_px": float(np.median(px[n])),
                             "rmse_px": float(np.sqrt(np.mean(np.square(px[n])))),
                             "median_pct_height": float(100 * np.median(rel[n])),
                             "p90_pct_height": float(100 * np.percentile(rel[n], 90))} for n in cfg.keypoints},
    }
    (args.out / "keypoints.json").write_text(json.dumps(report, indent=2))
    plots.keypoint_errors(rel, cfg.keypoints, args.out / "keypoint_errors.png")
    print(f"{matched}/{masks} masks matched, layout plausible for {report['plausible_layout']:.1%}  ->  {args.out}")


if __name__ == "__main__":
    main()
