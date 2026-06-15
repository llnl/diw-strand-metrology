#!/usr/bin/env python3
"""
Publication qualitative strip for the classical-baseline figure (Supplementary Fig. S7).

ONE shared FCT bleed-through image, shown as a 2-row strip:
  Row 1: raw | Otsu | Sauvola | Canny | Frangi | U-Net
         (raw = grayscale crop; each method = prediction mask)
  Row 2: human mask | error overlays for each method | U-Net error
         error overlay: green=TP, red=FP (over-segments lower-layer bleed-through),
         blue=FN (missed top-layer)

U-Net columns are a hatched PLACEHOLDER until the R4-full run lands; pass
--unet-mask <path to a U-Net pred png> to drop in the real prediction.

Default image chosen as a representative FCT case (all classical methods land in
their typical 0.54-0.70 Dice band -- not cherry-picked).
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import classical_baseline as cb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DEFAULT_IMG = "64b6cc5f67e3aa5bce6d5348-unproc.png"
METHODS = ["otsu", "sauvola", "canny", "frangi"]
MLABEL = {"otsu": "Otsu", "sauvola": "Sauvola", "canny": "Canny", "frangi": "Frangi"}


def error_rgb(pred, gt):
    e = np.zeros((*gt.shape, 3))
    e[pred & gt] = [0.1, 0.8, 0.1]
    e[pred & ~gt] = [0.9, 0.1, 0.1]
    e[~pred & gt] = [0.1, 0.3, 0.9]
    return e


def dice(pred, gt):
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else 2 * (pred & gt).sum() / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default=DEFAULT_IMG)
    ap.add_argument("--unet-mask", help="path to U-Net binary prediction PNG (1200x1200 or croppable)")
    args = ap.parse_args()

    frozen = json.load(open(os.path.join(OUT, "frozen_params.json")))
    for m in frozen:
        if "sigmas" in frozen[m]:
            frozen[m]["sigmas"] = np.array(frozen[m]["sigmas"])
    pairs = json.load(open(os.path.join(OUT, "sampled_pairs.json")))
    img2msk = {i: msk for i, msk in pairs["test"]}

    img_seg = cb.read_img(args.img)        # inverted (filaments bright)
    raw_disp = 1.0 - img_seg               # original polarity for display
    gt = cb.read_mask(img2msk[args.img])

    preds, errs, dices = {}, {}, {}
    for m in METHODS:
        p = cb.seg_apply(m, img_seg, frozen[m])
        preds[m] = p
        errs[m] = error_rgb(p, gt)
        dices[m] = dice(p, gt)

    # U-Net column
    unet_placeholder = args.unet_mask is None
    if not unet_placeholder:
        ua = np.asarray(Image.open(args.unet_mask))
        if ua.shape[:2] != gt.shape:
            ua = cb.crop_1200(ua)
        upred = ua > 0
        uerr = error_rgb(upred, gt)
        udice = dice(upred, gt)

    cols = ["raw"] + METHODS + ["unet"]
    fig, axes = plt.subplots(2, len(cols), figsize=(2.05 * len(cols), 4.4))

    # Row 0: raw, predictions
    axes[0, 0].imshow(raw_disp, cmap="gray")
    axes[0, 0].set_title("Raw (FCT)", fontsize=10)
    for j, m in enumerate(METHODS, start=1):
        axes[0, j].imshow(preds[m], cmap="gray")
        axes[0, j].set_title(f"{MLABEL[m]}\nDice={dices[m]:.2f}", fontsize=10)
    if unet_placeholder:
        axes[0, len(cols) - 1].add_patch(plt.Rectangle((0, 0), 1, 1, transform=axes[0, len(cols)-1].transAxes,
                                          facecolor="#f2d7d5", hatch="//", edgecolor="#c0392b"))
        axes[0, len(cols) - 1].text(0.5, 0.5, "U-Net\nprediction\n(pending\nR4 run)",
                                    ha="center", va="center", fontsize=9,
                                    transform=axes[0, len(cols) - 1].transAxes)
        axes[0, len(cols) - 1].set_title("U-Net", fontsize=10, color="#c0392b")
    else:
        axes[0, len(cols) - 1].imshow(upred, cmap="gray")
        axes[0, len(cols) - 1].set_title(f"U-Net\nDice={udice:.2f}", fontsize=10, color="#c0392b")

    # Row 1: human mask, error overlays.
    # pad=-2 nudges the title down toward its own image (away from row-0 above).
    axes[1, 0].imshow(gt, cmap="gray")
    axes[1, 0].set_title("Human mask", fontsize=10)
    for j, m in enumerate(METHODS, start=1):
        axes[1, j].imshow(errs[m])
    if unet_placeholder:
        axes[1, len(cols) - 1].add_patch(plt.Rectangle((0, 0), 1, 1, transform=axes[1, len(cols)-1].transAxes,
                                          facecolor="#f2d7d5", hatch="//", edgecolor="#c0392b"))
        axes[1, len(cols) - 1].text(0.5, 0.5, "(pending)", ha="center", va="center",
                                    fontsize=9, transform=axes[1, len(cols) - 1].transAxes)
    else:
        axes[1, len(cols) - 1].imshow(uerr)

    for a in axes.flat:
        a.set_xticks([]); a.set_yticks([])

    # left-side row labels
    axes[0, 0].set_ylabel("Prediction", fontsize=11)
    axes[1, 0].set_ylabel("Error vs. human", fontsize=11)

    fig.suptitle(
        "Why deep learning: classical methods over-segment lower-layer bleed-through\n"
        "Error overlay  —  green: correct (TP)   red: false positive / over-segment   blue: missed (FN)",
        fontsize=11, y=1.04)
    fig.tight_layout()
    # add vertical breathing room between the two rows so the row-1 titles never
    # collide with the row-0 images above them
    fig.subplots_adjust(hspace=0.32)
    out_png = os.path.join(OUT, "figS6_qual_strip.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"wrote {out_png} (+ .pdf)")
    print(f"image={args.img}  dices=" + ", ".join(f"{m}:{dices[m]:.3f}" for m in METHODS))


if __name__ == "__main__":
    main()
