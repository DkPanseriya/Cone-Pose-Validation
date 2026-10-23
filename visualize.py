"""Show how one cone is scored: the image, the FSOCO mask outline, and each predicted keypoint
with its distance to that outline.

    python visualize.py --model path/to/cone_pose.pt --image data/fsoco_segmentation_train/<team>/img/<name>
"""
import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from cone_validation import Config  # noqa: E402
from cone_validation.fsoco import load_cones  # noqa: E402
from cone_validation.keypoints import keypoint_errors, match_to_masks  # noqa: E402
from cone_validation.model import check_model, predict  # noqa: E402


def nearest_outline_point(point, contours):
    pts = np.vstack(contours)
    return pts[np.argmin(np.linalg.norm(pts - point, axis=1))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True, type=Path, help="an image from the FSOCO segmentation set")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--rank", type=int, default=0, help="0 = largest matched cone, 1 = second largest, ...")
    p.add_argument("--out", type=Path, default=Path("keypoint_check.png"))
    args = p.parse_args()

    cfg = Config.load(args.config)
    annotation = args.image.parent.parent / "ann" / (args.image.name + ".json")
    _, cones = load_cones(annotation, cfg)
    model = YOLO(args.model)
    check_model(model, cfg, need_keypoints=True)
    pred = predict(model, args.image, conf=args.conf, imgsz=args.imgsz)
    pairs = match_to_masks(pred.cls, pred.boxes, pred.scores, cones)
    if not pairs:
        raise SystemExit("no predicted cone matches a mask in this image")
    pairs.sort(key=lambda pc: -(cones[pc[1]].box[3] - cones[pc[1]].box[1]))
    p_idx, c_idx = pairs[min(args.rank, len(pairs) - 1)]
    cone, kpts = cones[c_idx], pred.keypoints[p_idx]
    err_px, err_rel = keypoint_errors(kpts, cone)

    image = cv2.cvtColor(cv2.imread(str(args.image)), cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = cone.box
    pad = 0.35 * (y2 - y1)
    xa, ya = int(max(x1 - pad, 0)), int(max(y1 - pad, 0))
    xb, yb = int(min(x2 + pad, image.shape[1])), int(min(y2 + pad, image.shape[0]))
    crop = image[ya:yb, xa:xb]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4))
    titles = ["camera image", "outline from the FSOCO mask", "predicted keypoints vs outline"]
    for ax, title in zip(axes, titles):
        ax.imshow(crop, extent=(xa, xb, yb, ya), alpha=1.0 if ax is axes[0] else 0.35)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[1:]:
        for c in cone.contours:
            loop = np.vstack([c, c[:1]])
            ax.plot(loop[:, 0], loop[:, 1], color="white" if ax is axes[1] else "#868e96", lw=2.5)
    ax = axes[2]
    worst = max(err_rel.max(), 1e-6)
    for i, (k, e) in enumerate(zip(kpts, err_rel)):
        q = nearest_outline_point(k, cone.contours)
        colour = plt.cm.RdYlGn_r(min(e / max(worst, 0.05), 1.0))
        ax.plot([k[0], q[0]], [k[1], q[1]], color="#d6336c", lw=1.5)
        ax.scatter(*k, s=70, color=colour, edgecolor="black", zorder=3)
        left = "left" in cfg.keypoints[i]
        ax.annotate(cfg.keypoints[i].replace("_", " "), k, xytext=(-8 if left else 8, 0), textcoords="offset points",
                    fontsize=8, ha="right" if left else "left", va="center")
    ax.text(0.02, 0.02, f"median error {100 * np.median(err_rel):.1f}% of cone height ({np.median(err_px):.1f} px)",
            transform=ax.transAxes, fontsize=9, color="black", backgroundcolor="white")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
