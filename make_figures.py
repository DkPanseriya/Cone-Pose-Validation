"""Explanatory figures for the README, from the FSOCO bounding-box set (no model needed).

    python make_figures.py --dataset data/fsoco_bounding_boxes_train
"""
import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cone_validation import Config  # noqa: E402
from cone_validation.distance import estimate_distance  # noqa: E402
from cone_validation.fsoco import load_cones, samples  # noqa: E402

OUT = Path("docs/figures")
CLASS_RGB = ["#e8590c", "#fd7e14", "#1c7ed6", "#f2c200"]


def variety_figure(dataset, n=8, seed=3):
    """One image from each of n teams: different cars, cameras, mountings, light."""
    rng = np.random.default_rng(seed)
    teams = sorted(d for d in dataset.iterdir() if (d / "img").is_dir())
    picks = rng.choice(len(teams), n, replace=False)
    fig, axes = plt.subplots(2, n // 2, figsize=(12, 4.2))
    for ax, k in zip(axes.ravel(), sorted(picks)):
        images = sorted((teams[k] / "img").iterdir())
        img = cv2.cvtColor(cv2.imread(str(images[len(images) // 2])), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        crop_w = min(w, int(h * 16 / 9))                           # same aspect for every tile
        x0 = (w - crop_w) // 2
        ax.imshow(img[:, x0:x0 + crop_w])
        ax.set_title(f"{teams[k].name}  ({w}x{h})", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"FSOCO: {len(teams)} teams, each with its own car, camera and mounting", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fsoco_variety.png", dpi=110)
    plt.close(fig)


def gate_figure(dataset, cfg):
    """One image: which labelled cones are scored and which are ignored, and why."""
    best = None
    for image, annotation in list(samples(dataset))[::25]:
        height, cones = load_cones(annotation, cfg)
        width = cv2.imread(str(image)).shape[1] if 6 <= len(cones) <= 16 else 0
        if not width or width / height > 1.8:        # cropped panoramas break the one-camera assumption
            continue
        d = estimate_distance([c.box[3] - c.box[1] for c in cones],
                              [cfg.heights_m[max(c.cls, 0)] for c in cones], height, cfg.vertical_fov_deg)
        reasons = []
        for c, dist in zip(cones, d):
            if c.cls < 0:
                reasons.append("unknown")
            elif c.tags & cfg.unreliable_tags:
                reasons.append(sorted(c.tags & cfg.unreliable_tags)[0].replace("_", " "))
            elif not cfg.gate_min_m <= dist <= cfg.gate_max_m:
                reasons.append("beyond 15 m" if dist > cfg.gate_max_m else "closer than 1 m")
            else:
                reasons.append("")
        kinds = {r for r in reasons if r}
        scored = sum(r == "" for r in reasons)
        score = len(kinds) * 10 + min(scored, 6) + 5 * ("beyond 15 m" in kinds)
        if best is None or score > best[0]:
            best = (score, image, cones, d, reasons)
    _, image, cones, d, reasons = best

    img = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
    boxes = np.array([c.box for c in cones])                      # crop to where the cones are
    x0, y0 = boxes[:, :2].min(0)
    x1, y1 = boxes[:, 2:].max(0)
    pad = 0.15 * max(x1 - x0, y1 - y0)
    x0, y0 = int(max(x0 - pad, 0)), int(max(y0 - pad, 0))
    x1, y1 = int(min(x1 + pad, img.shape[1])), int(min(y1 + pad, img.shape[0]))
    fig, ax = plt.subplots(figsize=(11, 11 * (y1 - y0) / (x1 - x0) + 0.6))
    ax.imshow(img[y0:y1, x0:x1], extent=(x0, x1, y1, y0))
    for c, dist, why in zip(cones, d, reasons):
        x1, y1, x2, y2 = c.box
        colour = "#2f9e44" if not why else "#adb5bd"
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=2 if not why else 1.2,
                                   color=colour, ls="-" if not why else "--"))
        if not why or y2 - y1 < 0.012 * img.shape[0]:  # label only the ignored cones, and only legible ones
            continue
        ax.text(x1, y1 - 4, why, fontsize=7.5, color="white",
                bbox={"facecolor": colour, "edgecolor": "none", "pad": 1.2})
    ax.axis("off")
    ax.set_title("Green: scored.  Grey: ignored, with the reason; a prediction on it counts neither for "
                 "nor against the model.", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "what_gets_scored.png", dpi=120)
    plt.close(fig)


def layout_figure(cfg):
    """Why outline distance needs a layout check: swapped keypoints still sit on the outline."""
    outline = np.array([[0, 0], [1.6, 0], [1.25, 1.9], [1.0, 3.2], [0.6, 3.2], [0.35, 1.9], [0, 0]])
    good = np.array([[0.1, 0.05], [0.25, 0.95], [0.45, 2.2], [0.62, 3.2],
                     [0.98, 3.2], [1.15, 2.2], [1.35, 0.95], [1.5, 0.05]])
    swapped = good[[0, 2, 1, 3, 4, 5, 6, 7]]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.8))
    for ax, k, title, ok in [(axes[0], good, "right order", True), (axes[1], swapped, "two keypoints swapped", False)]:
        ax.fill(outline[:, 0], outline[:, 1], color="#ffe8cc")
        ax.plot(outline[:, 0], outline[:, 1], color="#868e96", lw=2)
        ax.plot(*k[: len(k) // 2].T, color="#1c7ed6", lw=1, ls=":")
        ax.plot(*k[len(k) // 2:].T, color="#1c7ed6", lw=1, ls=":")
        ax.scatter(*k.T, s=60, color="#1c7ed6", edgecolor="black", zorder=3)
        for i, p in enumerate(k):
            ax.annotate(str(i), p, xytext=(-11 if i < 4 else 6, -3), textcoords="offset points", fontsize=8)
        ax.set_title(f"{title}\noutline error 0 px, layout {'passes' if ok else 'fails'}", fontsize=10,
                     color="#2b8a3e" if ok else "#c92a2a")
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle("Outline distance cannot see a swap, so the layout is checked separately", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "layout_check.png", dpi=150)
    plt.close(fig)


def hard_cases_figure(dataset, cfg, per_case=3, seed=0):
    """Cones that make FSOCO hard, and whether the validation scores them."""
    from cone_validation.metrics import box_iou

    # Picked by eye: geometry cannot tell lens distortion or an untagged fallen cone apart from others.
    picked = {
        "wide-angle lens": [("ff", "ff_00075.png", (1701, 1000, 1960, 1218)),
                            ("ff", "ff_00559.png", (1773, 846, 1887, 944)),
                            ("eufs", "eufs_00002.png", (195, 551, 273, 619))],
        "lying, not tagged": [("amz", "amz_00453.jpg", (1105, 1308, 1234, 1396)),
                              ("tuwr", "tuwr_00292.jpg", (5424, 2266, 6068, 2735)),
                              ("fsbizkaia", "fsbizkaia_00553.jpg", (2159, 1199, 2245, 1263))],
    }
    rng = np.random.default_rng(seed)
    pairs = list(samples(dataset))
    found = {k: [] for k in ["occluded", "dark", "far away", "truncated", "knocked over"]}
    used = set()
    for image, annotation in [pairs[i] for i in rng.permutation(len(pairs))]:
        if all(len(v) >= 40 for v in found.values()):
            break
        height, cones = load_cones(annotation, cfg)
        team = image.parent.parent.name
        boxes = np.array([c.box for c in cones]).reshape(-1, 4)
        ious = box_iou(boxes, boxes) if len(boxes) else np.zeros((0, 0))
        np.fill_diagonal(ious, 0)
        for k, c in enumerate(cones):
            h = c.box[3] - c.box[1]
            if c.cls < 0:
                continue
            if "truncated" in c.tags and h > 40:
                key = "truncated"
            elif "knocked_over" in c.tags and h > 25:
                key = "knocked over"
            elif not c.tags and h > 30 and ious[k].max() > 0.2:
                key = "occluded"
            elif not c.tags and h < 12:
                key = "far away"
            elif not c.tags and h > 40:
                key = "dark"
            else:
                continue
            if len(found[key]) < 40:
                found[key].append((image, c.box, team))

    def crop(image, box, size=150):
        img = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = box
        s = max(x2 - x1, y2 - y1) * 1.8 + 16
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        a, b = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0))
        c, d = int(min(cx + s / 2, img.shape[1])), int(min(cy + s / 2, img.shape[0]))
        tile = img[b:d, a:c].copy()
        cv2.rectangle(tile, (int(x1 - a), int(y1 - b)), (int(x2 - a), int(y2 - b)), (255, 255, 255), 1)
        return cv2.resize(tile, (size, size), interpolation=cv2.INTER_AREA)

    def brightness(cand):
        img = cv2.imread(str(cand[0]), cv2.IMREAD_GRAYSCALE)
        x1, y1, x2, y2 = (int(v) for v in cand[1])
        return img[y1:y2, x1:x2].mean()

    def pick(cands):
        chosen, teams = [], set()
        for cand in cands:
            if cand[2] not in teams and cand[0] not in used:
                chosen.append(cand)
                teams.add(cand[2])
                used.add(cand[0])
            if len(chosen) == per_case:
                break
        return chosen

    found["dark"] = sorted(found["dark"], key=brightness)
    columns = {
        "occluded": (pick(found["occluded"]), "scored"),
        "wide-angle lens": ([(dataset / t / "img" / n, b, t) for t, n, b in picked["wide-angle lens"]], "scored"),
        "dark": (pick(found["dark"]), "scored"),
        "lying, not tagged": ([(dataset / t / "img" / n, b, t) for t, n, b in picked["lying, not tagged"]],
                              "scored: no tag"),
        "far away": (pick(found["far away"]), "ignored: range gate"),
        "tagged truncated": (pick(found["truncated"]), "ignored: tag"),
        "tagged knocked over": (pick(found["knocked over"]), "ignored: tag"),
    }
    fig, axes = plt.subplots(per_case, len(columns), figsize=(13, 6.4))
    for j, (name, (chosen, verdict)) in enumerate(columns.items()):
        for i in range(per_case):
            ax = axes[i, j]
            ax.axis("off")
            if i < len(chosen):
                ax.imshow(crop(chosen[i][0], chosen[i][1]))
                ax.text(3, 145, chosen[i][2], fontsize=7, color="white",
                        bbox={"facecolor": "black", "alpha": 0.5, "pad": 1, "edgecolor": "none"})
        colour = "#2b8a3e" if verdict.startswith("scored") else "#868e96"
        axes[0, j].set_title(f"{name}\n{verdict}", fontsize=9.5, color=colour)
    fig.suptitle("What makes FSOCO hard, and what the validation scores", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "hard_cases.jpg", dpi=120, pil_kwargs={"quality": 88})
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, type=Path, help="FSOCO bounding-box root")
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(args.config)
    variety_figure(args.dataset)
    gate_figure(args.dataset, cfg)
    layout_figure(cfg)
    hard_cases_figure(args.dataset, cfg)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
