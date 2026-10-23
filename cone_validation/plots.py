"""Result figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CLASS_COLOURS = ["#e8590c", "#fd7e14", "#1c7ed6", "#f2c200"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def confusion(cm, names, path):
    labels = list(names) + ["background"]
    bg = len(labels) - 1
    share = cm / np.maximum(cm.sum(axis=0, keepdims=True), 1)       # each ground-truth column sums to 1
    share[:, bg] = 0                                                  # false positives: counts only
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == bg and j == bg:
                continue
            text = f"{cm[i, j]}" if j == bg else f"{share[i, j]:.2f}\n({cm[i, j]})"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if share[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("ground truth")
    ax.set_ylabel("predicted")
    ax.set_title("Confusion matrix (share of each ground-truth class)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ap_vs_iou(ap, thresholds, names, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for c, name in enumerate(names):
        if not np.isnan(ap[c]).all():
            ax.plot(thresholds, ap[c], marker="o", ms=4, color=CLASS_COLOURS[c % 4], label=name)
    ax.plot(thresholds, np.nanmean(ap, axis=0), color="black", lw=2.5, label="mean")
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("average precision")
    ax.set_ylim(0, 1)
    ax.set_title("How AP falls as box overlap must get tighter")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def recall_by_distance(bins, detected, correct, counts, path):
    centres = np.arange(len(bins) - 1)
    labels = [f"{bins[i]:g}-{bins[i + 1]:g} m\n({n} cones)" for i, n in enumerate(counts)]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(centres - 0.2, detected, 0.4, color="#4c6ef5", label="detected")
    ax.bar(centres + 0.2, correct, 0.4, color="#94d82d", label="detected, right colour")
    ax.set_xticks(centres, labels)
    ax.set_ylim(0, 1.25)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("recall")
    ax.set_title("Recall by estimated distance")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def keypoint_errors(errors, names, path):
    """Box plot of outline distance per keypoint, as % of cone height."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    data = [100 * np.asarray(errors[n]) for n in names]
    ax.boxplot(data, vert=False, showfliers=False, widths=0.6,
               medianprops={"color": "#e8590c", "lw": 2}, boxprops={"color": "#333"})
    ax.set_yticks(range(1, len(names) + 1), [n.replace("_", " ") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("distance to mask outline, % of cone height")
    ax.set_title("Keypoint error per keypoint")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def error_by_size(rows, path, bins=(0, 20, 40, 80, 160, 10000)):
    """Median keypoint error per cone against the cone's height in pixels."""
    height, err = rows[:, 0], 100 * rows[:, 1]
    labels, med, lo, hi, counts = [], [], [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        e = err[(height >= a) & (height < b)]
        if len(e) < 5:
            continue
        labels.append(f"{a}-{b} px" if b < 10000 else f">{a} px")
        med.append(np.median(e))
        lo.append(np.percentile(e, 25))
        hi.append(np.percentile(e, 75))
        counts.append(len(e))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(x, med, color="#4c6ef5", width=0.6)
    ax.errorbar(x, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="none", color="#333", capsize=4)
    ax.set_xticks(x, [f"{label}\n({n} cones)" for label, n in zip(labels, counts)])
    ax.set_xlabel("cone height in the image")
    ax.set_ylabel("keypoint error, % of cone height")
    ax.set_title("Keypoint error against cone size (median, middle 50%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
