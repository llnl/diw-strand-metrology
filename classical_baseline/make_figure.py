#!/usr/bin/env python3
"""
Build the classical-baseline figure (Supplementary Fig. S6): classical-CV segmentation
baselines vs U-Net.

Panel A: grouped bar chart of mean Dice by geometry (SC/FCT/HELI), bars = the
         4 classical methods + U-Net (highlighted). Error bars = 95% CI of the
         per-image mean.
Table  : method x {SC, FCT, HELI, ALL} mean Dice, written as CSV + LaTeX.

U-Net per-geometry test Dice is a PLACEHOLDER until the R4-full MLflow run lands
(test_dice_Structure_{SC,FCT,HELI} in experiment diw-R4-ablation). Override by
editing UNET_DICE below, or pass --unet "SC,FCT,HELI,ALL".

Usage:
    python3 make_figure.py                       # uses placeholder U-Net bar
    python3 make_figure.py --unet 0.972,0.979,0.974,0.977
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CSV_PATH = os.path.join(OUT, "per_image_dice.csv")

GEOMS = ["SC", "FCT", "HELI"]
# Display order left->right; U-Net last so it sits at the right of each group.
METHODS = ["otsu", "sauvola", "canny", "frangi", "unet"]
LABELS = {
    "otsu": "Otsu\n(threshold)",
    "sauvola": "Sauvola\n(adaptive)",
    "canny": "Canny\n(edges)",
    "frangi": "Frangi\n(ridge)",
    "unet": "U-Net",
}
# Greys for classical; the paper's accent colour for U-Net.
COLORS = {
    "otsu": "#bdbdbd", "sauvola": "#9e9e9e",
    "canny": "#cfcfcf", "frangi": "#7d7d7d",
    "unet": "#c0392b",
}

# PLACEHOLDER U-Net per-geometry test Dice. Replace with R4-full MLflow
# test_dice_Structure_{SC,FCT,HELI}; ALL ~ corpus headline 0.977.
UNET_DICE = {"SC": 0.977, "FCT": 0.977, "HELI": 0.977, "ALL": 0.977}
UNET_IS_PLACEHOLDER = True


def load_rows():
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def stats(rows):
    """Return {method: {geom: (mean, ci95, n)}} plus ALL, for classical methods."""
    out = {}
    for m in ["otsu", "sauvola", "canny", "frangi"]:
        out[m] = {}
        allv = []
        for g in GEOMS:
            v = np.array([float(r["dice"]) for r in rows
                          if r["method"] == m and r["geom"] == g])
            allv.append(v)
            ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            out[m][g] = (float(v.mean()), float(ci), len(v))
        av = np.concatenate(allv)
        ci = 1.96 * av.std(ddof=1) / np.sqrt(len(av))
        out[m]["ALL"] = (float(av.mean()), float(ci), len(av))
    return out


def make_bar(st, unet, fig_path):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    n_m = len(METHODS)
    group_w = 0.8
    bar_w = group_w / n_m
    x = np.arange(len(GEOMS))

    for j, m in enumerate(METHODS):
        offs = (j - (n_m - 1) / 2) * bar_w
        if m == "unet":
            means = [unet[g] for g in GEOMS]
            errs = [0.0] * len(GEOMS)
        else:
            means = [st[m][g][0] for g in GEOMS]
            errs = [st[m][g][1] for g in GEOMS]
        bars = ax.bar(x + offs, means, bar_w * 0.92, yerr=errs, capsize=2.5,
                      color=COLORS[m], edgecolor="black", linewidth=0.6,
                      label=LABELS[m].replace("\n", " "),
                      hatch="//" if (m == "unet" and UNET_IS_PLACEHOLDER) else None,
                      error_kw=dict(lw=0.8))
        # value labels
        for b, mn in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, mn + 0.012, f"{mn:.2f}",
                    ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(GEOMS, fontsize=12)
    ax.set_ylabel("Dice coefficient", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("Lattice geometry", fontsize=12)
    title = ("Segmentation accuracy: classical CV vs. U-Net\n"
             "(per-image Dice on the held-out test set, "
             "classical knobs tuned on val & frozen)")
    ax.set_title(title, fontsize=11)

    handles, labels = ax.get_legend_handles_labels()
    if UNET_IS_PLACEHOLDER:
        handles.append(Patch(facecolor=COLORS["unet"], hatch="//",
                             edgecolor="black",
                             label="U-Net (PLACEHOLDER – awaiting R4 run)"))
        labels.append("U-Net (PLACEHOLDER – awaiting R4 run)")
        handles = handles[:-2]; labels = labels[:-2]  # drop dup u-net entry
        handles.append(Patch(facecolor=COLORS["unet"], hatch="//",
                             edgecolor="black"))
        labels.append("U-Net (PLACEHOLDER)")
    ax.legend(handles, labels, fontsize=8.5, ncol=5, loc="lower center",
              bbox_to_anchor=(0.5, -0.28), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"wrote {fig_path} (+ .pdf)")


def write_table(st, unet):
    cols = GEOMS + ["ALL"]
    # CSV
    csv_path = os.path.join(OUT, "dice_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + cols)
        for m in ["otsu", "sauvola", "canny", "frangi"]:
            w.writerow([m] + [f"{st[m][c][0]:.4f}" for c in cols])
        w.writerow(["unet"] + [f"{unet[c]:.4f}" + ("*" if UNET_IS_PLACEHOLDER else "")
                               for c in cols])
    # LaTeX
    tex_path = os.path.join(OUT, "dice_table.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n")
        f.write("Method & " + " & ".join(cols) + " \\\\\n\\midrule\n")
        names = {"otsu": "Otsu (threshold)", "sauvola": "Sauvola (adaptive)",
                 "canny": "Canny (edges)", "frangi": "Frangi (ridge)"}
        for m in ["otsu", "sauvola", "canny", "frangi"]:
            f.write(names[m] + " & " + " & ".join(f"{st[m][c][0]:.3f}" for c in cols)
                    + " \\\\\n")
        f.write("\\midrule\n")
        star = "$^{*}$" if UNET_IS_PLACEHOLDER else ""
        f.write("\\textbf{U-Net}" + star + " & "
                + " & ".join(f"\\textbf{{{unet[c]:.3f}}}" for c in cols) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {csv_path} and {tex_path}")


def main():
    global UNET_IS_PLACEHOLDER
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", help="real U-Net Dice as 'SC,FCT,HELI,ALL'")
    args = ap.parse_args()

    rows = load_rows()
    st = stats(rows)

    unet = dict(UNET_DICE)
    if args.unet:
        sc, fct, heli, allv = [float(x) for x in args.unet.split(",")]
        unet = {"SC": sc, "FCT": fct, "HELI": heli, "ALL": allv}
        UNET_IS_PLACEHOLDER = False

    # console summary
    print(f"\n{'method':<10}" + "".join(f"{c:>9}" for c in GEOMS + ['ALL']))
    for m in ["otsu", "sauvola", "canny", "frangi"]:
        print(f"{m:<10}" + "".join(f"{st[m][c][0]:>9.3f}" for c in GEOMS + ['ALL']))
    tag = " (PLACEHOLDER)" if UNET_IS_PLACEHOLDER else ""
    print(f"{'unet':<10}" + "".join(f"{unet[c]:>9.3f}" for c in GEOMS + ['ALL']) + tag)

    make_bar(st, unet, os.path.join(OUT, "figS6_dice_bar.png"))
    write_table(st, unet)


if __name__ == "__main__":
    main()
