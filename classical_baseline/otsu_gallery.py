#!/usr/bin/env python3
"""
Render a gallery of Otsu segmentation results so we can eyeball where it
succeeds and (especially) where it fails.

For each sampled image, 4 columns:
  raw (cropped, original polarity) | human mask | Otsu prediction | error overlay
Error overlay: green = correct foreground, red = false positive (Otsu over-segments,
e.g. lower-layer bleed-through), blue = false negative (missed top-layer).

Samples: per geometry, the best / median / worst Otsu-Dice test images, so the
range is visible rather than cherry-picked. Uses the FROZEN Otsu params.
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import classical_baseline as cb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
GEOMS = ["SC", "FCT", "HELI"]


def main():
    rows = list(csv.DictReader(open(os.path.join(OUT, "per_image_dice.csv"))))
    otsu = [r for r in rows if r["method"] == "otsu"]
    params = json.load(open(os.path.join(OUT, "frozen_params.json")))["otsu"]
    # map img filename -> mask filename from sampled pairs
    pairs = json.load(open(os.path.join(OUT, "sampled_pairs.json")))
    img2msk = {i: m for i, m in pairs["test"]}

    # pick best/median/worst per geometry
    picks = []  # (geom, tag, img, dice)
    for g in GEOMS:
        gr = sorted([r for r in otsu if r["geom"] == g], key=lambda r: float(r["dice"]))
        if not gr:
            continue
        idxs = {"worst": 0, "median": len(gr) // 2, "best": len(gr) - 1}
        for tag, ix in idxs.items():
            picks.append((g, tag, gr[ix]["img"], float(gr[ix]["dice"])))

    n = len(picks)
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.0 * n))
    col_titles = ["Raw (1200 crop)", "Human mask",
                  "Otsu prediction", "Error: G=TP R=FP(over) B=FN"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=11)

    for i, (g, tag, img_fn, d) in enumerate(picks):
        msk_fn = img2msk[img_fn]
        # raw for display: original polarity (filaments dark) -> show 1-inverted
        raw_disp = 1.0 - cb.read_img(img_fn)  # read_img inverts; undo for display
        gt = cb.read_mask(msk_fn)
        pred = cb.seg_apply("otsu", cb.read_img(img_fn), params)

        tp = pred & gt
        fp = pred & ~gt
        fn = ~pred & gt
        err = np.zeros((*gt.shape, 3))
        err[tp] = [0.1, 0.8, 0.1]
        err[fp] = [0.9, 0.1, 0.1]
        err[fn] = [0.1, 0.3, 0.9]

        axes[i, 0].imshow(raw_disp, cmap="gray")
        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 3].imshow(err)
        axes[i, 0].set_ylabel(f"{g} · {tag}\nDice={d:.3f}", fontsize=10)
        for j in range(4):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

    fp_frac = np.mean([
        (lambda p, gt: (p & ~gt).sum() / max(p.sum(), 1))(
            cb.seg_apply("otsu", cb.read_img(im), params), cb.read_mask(img2msk[im]))
        for _, _, im, _ in picks])
    fig.suptitle(
        f"Otsu segmentation gallery (frozen params; mean FP/pred over shown = {fp_frac:.2f})\n"
        "Red = Otsu marks filament where human did NOT (lower-layer bleed-through over-segmentation)",
        fontsize=12, y=1.005)
    fig.tight_layout()
    out_png = os.path.join(OUT, "otsu_gallery.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"wrote {out_png}")
    for g, tag, im, d in picks:
        print(f"  {g:5s} {tag:7s} Dice={d:.3f}  {im}")


if __name__ == "__main__":
    main()
