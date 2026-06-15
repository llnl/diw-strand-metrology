#!/usr/bin/env python3
"""
Dump ~50 segmentation results per method (otsu, sauvola, canny, frangi) so we
can browse where each succeeds/fails.

For each method:
  - out/galleries/<method>/  : 50 individual 4-panel PNGs, named
        {rank:02d}_{dice:.3f}_{geom}.png  (sortable by Dice)
        panels = raw (1200 crop) | human mask | prediction | error overlay
        error overlay: green=TP, red=FP(over-seg), blue=FN(missed)
  - out/galleries/<method>_contact.png : single contact sheet (10x5) of the
        error overlays for a quick at-a-glance view.

Sampling: 50 images per method, evenly spaced along the Dice-sorted test set
(so the full range from worst to best is represented), stratified to keep the
geometry mix. Uses FROZEN params; reuses cached test images.
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
GAL = os.path.join(OUT, "galleries")
METHODS = ["otsu", "sauvola", "canny", "frangi"]
N_PER = 50


def even_sample(sorted_rows, n):
    """Pick n rows evenly spaced along a sorted list (covers full range)."""
    if len(sorted_rows) <= n:
        return sorted_rows
    idx = np.linspace(0, len(sorted_rows) - 1, n).round().astype(int)
    return [sorted_rows[i] for i in idx]


def error_rgb(pred, gt):
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    e = np.zeros((*gt.shape, 3))
    e[tp] = [0.1, 0.8, 0.1]
    e[fp] = [0.9, 0.1, 0.1]
    e[fn] = [0.1, 0.3, 0.9]
    return e


def main():
    rows = list(csv.DictReader(open(os.path.join(OUT, "per_image_dice.csv"))))
    frozen = json.load(open(os.path.join(OUT, "frozen_params.json")))
    for m in frozen:
        if "sigmas" in frozen[m]:
            frozen[m]["sigmas"] = np.array(frozen[m]["sigmas"])
    pairs = json.load(open(os.path.join(OUT, "sampled_pairs.json")))
    img2msk = {i: msk for i, msk in pairs["test"]}

    os.makedirs(GAL, exist_ok=True)

    for method in METHODS:
        mr = sorted([r for r in rows if r["method"] == method],
                    key=lambda r: float(r["dice"]))
        sample = even_sample(mr, N_PER)
        mdir = os.path.join(GAL, method)
        os.makedirs(mdir, exist_ok=True)

        errs = []  # for contact sheet
        for rank, r in enumerate(sample):
            img_fn = r["img"]
            d = float(r["dice"])
            g = r["geom"]
            img_seg = cb.read_img(img_fn)             # inverted (filaments bright)
            raw_disp = 1.0 - img_seg                   # original polarity for display
            gt = cb.read_mask(img2msk[img_fn])
            pred = cb.seg_apply(method, img_seg, frozen[method])
            e = error_rgb(pred, gt)
            errs.append((e, d, g))

            fig, ax = plt.subplots(1, 4, figsize=(12, 3.1))
            ax[0].imshow(raw_disp, cmap="gray"); ax[0].set_title("raw", fontsize=9)
            ax[1].imshow(gt, cmap="gray"); ax[1].set_title("human", fontsize=9)
            ax[2].imshow(pred, cmap="gray"); ax[2].set_title(f"{method}", fontsize=9)
            ax[3].imshow(e); ax[3].set_title("G=TP R=FP B=FN", fontsize=9)
            for a in ax:
                a.set_xticks([]); a.set_yticks([])
            fig.suptitle(f"{method}  {g}  Dice={d:.3f}", fontsize=11, y=1.02)
            fig.tight_layout()
            fig.savefig(os.path.join(mdir, f"{rank:02d}_{d:.3f}_{g}.png"),
                        dpi=90, bbox_inches="tight")
            plt.close(fig)

        # contact sheet: 10 cols x 5 rows of error overlays
        cols, rrows = 10, 5
        fig, axes = plt.subplots(rrows, cols, figsize=(cols * 1.5, rrows * 1.6))
        for k, ax in enumerate(axes.flat):
            if k < len(errs):
                e, d, g = errs[k]
                ax.imshow(e)
                ax.set_title(f"{g} {d:.2f}", fontsize=6.5)
            ax.set_xticks([]); ax.set_yticks([])
        mean_d = np.mean([float(r["dice"]) for r in mr])
        fig.suptitle(f"{method.upper()} — {len(errs)} results across full Dice range "
                     f"(mean ALL={mean_d:.3f}). Red=over-seg, Blue=missed.",
                     fontsize=12, y=1.005)
        fig.tight_layout()
        fig.savefig(os.path.join(GAL, f"{method}_contact.png"),
                    dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  {method}: {len(sample)} PNGs -> {mdir}/  + {method}_contact.png")

    print(f"\nAll galleries under: {GAL}")


if __name__ == "__main__":
    main()
