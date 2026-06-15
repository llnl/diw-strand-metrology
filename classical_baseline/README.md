# Classical computer-vision baseline

This directory reproduces the classical-segmentation baseline reported in the paper (necessity of a
learned model; Supplementary Table S3 and Figs. S6–S7). It runs a ladder of deterministic classical
segmenters on the **same** held-out test split used to evaluate the U-Net, with each method's global
hyperparameters tuned on the validation split and frozen before a single evaluation on the test split
(one global configuration per method, mirroring the single global U-Net).

## Methods

All from scikit-image, no learned weights:

- **Otsu** and **Sauvola** — global / adaptive thresholding
- **Canny** — edge detection with morphological closing
- **Frangi** — vesselness (ridge) filtering, a stronger baseline since filaments are tubular ridges

Filaments are dark, so each image is intensity-inverted once before processing (a fixed preprocessing
choice, not a tuned knob).

## Protocol

Tune each method's global hyperparameters on the validation split, freeze them, then evaluate once on the
disjoint test split. Inputs are identical to the U-Net: the fixed 1200×1200 validation/test crop applied
to the 2048×2048 source images. Dice is computed per image and averaged.

## Usage

```shell
# 1) compute per-image Dice on the test split (downloads + caches images/masks from S3)
python classical_baseline.py            # writes out/per_image_dice.csv, out/dice_summary.json, out/frozen_params.json

# 2) grouped-bar figure + Dice table (pass the U-Net per-geometry numbers)
python make_figure.py --unet 0.974,0.972,0.975,0.973   # SC,FCT,HELI,ALL -> out/figS6_dice_bar.{png,pdf}, out/dice_table.{csv,tex}

# 3) optional: per-method galleries and the qualitative error strip
python method_galleries.py
python qual_strip.py --unet-mask <path to a U-Net prediction PNG>
```

## Data

Images/masks and the split/geometry files are the same as the main training pipeline:

- Split file: `diw_R4_full.json` (train/val/test; the leak-free stratified 80/10/10 split)
- Geometry labels: `diw_R4_geom.csv` (columns `S3 URI`, `Structure` with values SC/FCT/HELI)
- Images/masks: streamed from the project S3 bucket and cached locally under `cache/`.

## Result (summary)

Best classical method (Otsu) reaches mean Dice 0.708 versus 0.973 for the U-Net; classical accuracy is
worst on the dense FCT lattice (Canny 0.497), where lower-layer bleed-through is most prevalent, whereas
the U-Net is flat across geometries (0.972–0.975). Widening the tuning grid ~9× does not close the gap,
confirming a performance ceiling rather than under-tuning. The errors are dominated by false positives in
which lower-layer or inter-filament structure is labeled as top-layer filament — a semantic distinction
inaccessible to intensity- or edge-based rules.
