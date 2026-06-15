#!/usr/bin/env python3
"""
Classical-CV segmentation baseline for the DIW on-machine inspection pipeline.

Demonstrates the necessity of a learned model by running a ladder of deterministic
classical segmenters on the SAME split the U-Net trained on, scoring Dice against
the human top-layer masks.

Protocol (fair tune-and-freeze):
  - Tune each method's GLOBAL hyperparameters on the train/val pool.
  - Freeze them. Evaluate ONCE on the frozen test split (1,380 imgs).
  - One global knob set per method (not per-geometry) to mirror the single U-Net.

Methods (all scikit-image, no learned weights):
  otsu     - global Otsu threshold + Gaussian pre-blur + morphological cleanup (thresholding)
  sauvola  - adaptive/local threshold (handles uneven LED-ring illumination)
  canny    - Canny edges + morphological closing + fill (edge detection)
  frangi   - Frangi vesselness ridge filter + threshold
             (the honest strong baseline: filaments are tubular ridges)

The scientific point: classical methods key off intensity/edges, but in a
log-pile lattice the lower-layer bleed-through has nearly identical brightness
statistics to the top layer. Separating them is a SEMANTIC task no threshold or
edge rule can do -- which is exactly where these methods over-segment and U-Net
does not.

Data source: s3://baselabelinginfrastack-alllabeledimages (account 159929462505,
us-west-2). Images are 2048x2048 16-bit (I;16); masks 2048x2048 binary {0,1}.
Dice is scored at native resolution. Pixel pitch (3.249 um/px) is NOT needed here
-- it only matters for the deferred diameter step.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
from PIL import Image
from skimage.filters import (
    threshold_otsu,
    threshold_sauvola,
    frangi,
    gaussian,
)
from skimage.feature import canny
from skimage.morphology import (
    binary_closing,
    binary_opening,
    remove_small_objects,
    remove_small_holes,
    disk,
)
from scipy import ndimage as ndi

# ----------------------------------------------------------------------------
# Paths / constants
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
BUCKET = "baselabelinginfrastack-alllabeledimages"
IMG_PREFIX = "images_camp1_cln_camp2_1176_camp2_218_camp2_25_camp3_14"
MSK_PREFIX = "masks_camp1_cln_camp2_1176_camp2_218_camp2_25_camp3_14"
REGION = "us-west-2"

SPLIT_JSON = os.path.join(HERE, "diw_R4_full.json")
GEOM_CSV = os.path.join(HERE, "diw_R4_geom.csv")
IMG_CACHE = os.path.join(HERE, "cache", "img")
MSK_CACHE = os.path.join(HERE, "cache", "msk")
OUT_DIR = os.path.join(HERE, "out")

GEOMS = ["SC", "FCT", "HELI"]
METHODS = ["otsu", "sauvola", "canny", "frangi"]

# The U-Net was scored on a FIXED 1200x1200 crop, NOT native 2048. Both val and
# test datasets use the val_transform, whose first op in the champion's
# default_transforms.yaml is:
#     Crop(x_min=374, x_max=1574, y_min=364, y_max=1564)  ->  1200x1200
# (verified in code: builder.py:53-54 test uses val_transform; default_transforms.yaml).
# Classical methods MUST see the identical inputs, so we apply the same crop to
# both image and mask. NOTE: this is a crop, not a resize -- filament pixel
# scale is unchanged, so Frangi sigmas (set from nozzle px radius) stay valid.
CROP_Y = slice(364, 1564)
CROP_X = slice(374, 1574)


def crop_1200(a):
    """Apply the U-Net val/test 1200x1200 crop window."""
    return a[CROP_Y, CROP_X]


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_geom():
    with open(GEOM_CSV) as f:
        return {r["S3 URI"]: r["Structure"] for r in csv.DictReader(f)}


def load_split():
    with open(SPLIT_JSON) as f:
        return json.load(f)


def split_items(split_name):
    """Return list of (img_filename, mask_filename) for a split."""
    d = load_split()
    s = d[split_name]
    return list(zip(s["image"], s["mask"]))


def stratified_sample(split_name, per_geom, seed=42):
    """Pick `per_geom` images of each geometry from a split (reproducible)."""
    geom = load_geom()
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for img, msk in split_items(split_name):
        buckets[geom.get(img, "??")].append((img, msk))
    picked = []
    for g in GEOMS:
        items = buckets[g]
        idx = rng.permutation(len(items))[:per_geom]
        picked += [items[i] for i in idx]
    return picked


def _parallel_cp(srcs_dsts, jobs=32):
    """Download many s3 objects concurrently via a thread pool of `aws s3 cp`.

    Per-file `aws` CLI startup (~0.7s) dominates a serial loop; running 32 in
    parallel cuts a ~1,500-file pull from ~18 min to a few minutes.
    """
    if not srcs_dsts:
        return
    from concurrent.futures import ThreadPoolExecutor

    def cp(sd):
        s, d = sd
        r = subprocess.run(
            ["aws", "s3", "cp", s, d, "--region", REGION, "--quiet"],
            capture_output=True, text=True,
        )
        return None if r.returncode == 0 else (s, r.stderr.strip())

    fails = []
    done = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for res in ex.map(cp, srcs_dsts):
            done += 1
            if res is not None:
                fails.append(res)
            if done % 200 == 0:
                print(f"    {done}/{len(srcs_dsts)} files...", flush=True)
    if fails:
        print(f"  WARNING: {len(fails)} downloads failed; first: {fails[0]}")


def s3_download(pairs):
    """Download (img, mask) pairs into cache if not already present."""
    os.makedirs(IMG_CACHE, exist_ok=True)
    os.makedirs(MSK_CACHE, exist_ok=True)
    todo = []
    for i, m in pairs:
        di = os.path.join(IMG_CACHE, i)
        dm = os.path.join(MSK_CACHE, m)
        if not os.path.exists(di):
            todo.append((f"s3://{BUCKET}/{IMG_PREFIX}/{i}", di))
        if not os.path.exists(dm):
            todo.append((f"s3://{BUCKET}/{MSK_PREFIX}/{m}", dm))
    print(f"  need {len(todo)} files (of {len(pairs)} pairs) -> parallel cp")
    _parallel_cp(todo)


def read_img(fn):
    """16-bit image -> 1200x1200 crop -> float32 in [0,1], filaments made BRIGHT.

    Top-layer filaments are DARKER than background (verified: mean intensity
    inside the human mask ~0.21 vs ~0.40 outside). We invert once so that
    "bright = filament" holds for all threshold/ridge segmenters. This is a
    fixed, global preprocessing choice (not a per-image tuned knob).
    """
    a = np.asarray(Image.open(os.path.join(IMG_CACHE, fn))).astype(np.float32)
    a = crop_1200(a)
    lo, hi = np.percentile(a, [0.5, 99.5])  # robust stretch on the crop
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    return 1.0 - a  # invert: dark filaments -> bright foreground


def read_mask(fn):
    a = np.asarray(Image.open(os.path.join(MSK_CACHE, fn)))
    return (crop_1200(a) > 0).astype(bool)


# ----------------------------------------------------------------------------
# Classical segmenters, split into an EXPENSIVE base + a CHEAP finalize so a
# wide tuning grid stays fast: the base (Frangi vesselness, Sauvola threshold
# map, Canny edges, Gaussian blur) is computed ONCE per image and the cheap
# threshold/morphology sweeps reuse it (see tune_all).
#
# A "param dict" carries BOTH base keys and finalize keys. BASE_FUNCS read only
# their base keys; FIN_FUNCS read only their finalize keys.
# ----------------------------------------------------------------------------
def _cleanup(mask, p):
    if p.get("close_r"):
        mask = binary_closing(mask, disk(p["close_r"]))
    if p.get("open_r"):
        mask = binary_opening(mask, disk(p["open_r"]))
    if p.get("min_hole"):
        mask = remove_small_holes(mask, area_threshold=p["min_hole"])
    if p.get("min_obj"):
        mask = remove_small_objects(mask, min_size=p["min_obj"])
    return mask


# --- bases (expensive) ---
def base_otsu(img, p):
    return gaussian(img, sigma=p["blur_sigma"])  # blurred grayscale


def base_sauvola(img, p):
    b = gaussian(img, sigma=p["blur_sigma"])
    t = threshold_sauvola(b, window_size=p["window"], k=p["k"])
    return b > t  # binary


def base_canny(img, p):
    return canny(img, sigma=p["sigma"], low_threshold=p["low"], high_threshold=p["high"])


def base_frangi(img, p):
    # Filaments are bright tubular ridges. Scales (sigma ~ filament radius in px)
    # span the working regime: diameters ~240-300 um / 3.249 um/px -> radius
    # ~37-46 px; the grid brackets the full 0.15-0.50 mm nozzle range. Crop is
    # 1:1 (no resize) so px scale matches native; sigmas are valid as-is.
    resp = frangi(img, sigmas=p["sigmas"], black_ridges=False)
    m = resp.max()
    return resp / m if m > 0 else resp


# --- finalizers (cheap; operate on cached base) ---
def fin_otsu(base, p):
    m = base > (threshold_otsu(base) + p.get("offset", 0.0))
    return _cleanup(m, p)


def fin_sauvola(base, p):
    return _cleanup(base, p)


def fin_canny(base, p):
    m = binary_closing(base, disk(p["close_r"]))
    m = ndi.binary_fill_holes(m)
    return _cleanup(m, p)


def fin_frangi(base, p):
    return _cleanup(base > p["resp_thresh"], p)


BASE_FUNCS = {"otsu": base_otsu, "sauvola": base_sauvola,
              "canny": base_canny, "frangi": base_frangi}
FIN_FUNCS = {"otsu": fin_otsu, "sauvola": fin_sauvola,
             "canny": fin_canny, "frangi": fin_frangi}

# Which keys define the expensive base (used to cache base across cheap sweeps).
BASE_KEYS = {"otsu": ("blur_sigma",),
             "sauvola": ("blur_sigma", "window", "k"),
             "canny": ("sigma", "low", "high"),
             "frangi": ("sigmas",)}


def seg_apply(method, img, p):
    """Full segment (base + finalize) for a merged param dict. Used in eval."""
    return FIN_FUNCS[method](BASE_FUNCS[method](img, p), p)


def base_sig(method, p):
    """Hashable signature of the base-determining params (for caching)."""
    return tuple(tuple(p[k]) if k == "sigmas" else p[k] for k in BASE_KEYS[method])


# ----------------------------------------------------------------------------
# Widened tuning grids. Built as (base x threshold x morphology) so each method
# explores its core knobs broadly while morphology stays a small shared sweep.
# Tuned on val, frozen, evaluated once on test (same as before).
# ----------------------------------------------------------------------------
# Shared morphology post-processing options (cheap; applied after thresholding).
_MORPH = [
    {"open_r": 1, "close_r": 2, "min_obj": 2000, "min_hole": 2000},
    {"open_r": 2, "close_r": 3, "min_obj": 3000, "min_hole": 3000},
    {"open_r": 0, "close_r": 1, "min_obj": 1000, "min_hole": 1000},
]


def _grid(base_dicts, sweep_dicts):
    """Cartesian product of base configs x (threshold/morph) configs -> merged."""
    out = []
    for b in base_dicts:
        for s in sweep_dicts:
            out.append({**b, **s})
    return out


PARAM_GRID = {
    # threshold offset widens the Otsu cut (under/over-segment) around the auto value
    "otsu": _grid(
        [{"blur_sigma": s} for s in (0, 1, 2, 3)],
        [{**m, "offset": o} for o in (-0.06, -0.03, 0.0, 0.03, 0.06) for m in _MORPH],
    ),
    "sauvola": _grid(
        [{"blur_sigma": 1, "window": w, "k": k}
         for w in (25, 51, 75, 101, 151, 201) for k in (0.05, 0.1, 0.15, 0.2, 0.3)],
        _MORPH,
    ),
    "canny": _grid(
        [{"sigma": s, "low": lo, "high": hi}
         for s in (1, 2, 3)
         for (lo, hi) in ((0.05, 0.15), (0.1, 0.2), (0.15, 0.3), (0.2, 0.4))],
        [{**m, "close_r": c} for c in (3, 5, 7)
         for m in ({"min_obj": 2000, "min_hole": 5000},
                   {"min_obj": 1000, "min_hole": 10000})],
    ),
    "frangi": _grid(
        [{"sigmas": sg} for sg in (
            np.arange(8, 40, 8), np.arange(12, 60, 12),
            np.arange(20, 80, 15), np.arange(30, 50, 5),
            np.arange(6, 30, 6))],
        [{**m, "resp_thresh": t}
         for t in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3) for m in _MORPH[:2]],
    ),
}


# ----------------------------------------------------------------------------
# Metric
# ----------------------------------------------------------------------------
def dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else 2.0 * inter / s


# ----------------------------------------------------------------------------
# Parallel workers (module-level so they pickle for multiprocessing). Each loads
# its image/mask by filename, so only small tuples cross process boundaries.
# ----------------------------------------------------------------------------
def _tune_worker(task):
    """For one (method, image): compute each distinct base ONCE, then sweep all
    grid params over the cached base. Returns Dice for every grid index.

    This is the speedup that makes a WIDE grid cheap -- e.g. all Frangi
    threshold/morph variants reuse one vesselness response per image.
    """
    method, img_fn, msk_fn = task
    img = read_img(img_fn)
    gt = read_mask(msk_fn)
    grid = PARAM_GRID[method]
    base_cache = {}
    out = []
    for pidx, p in enumerate(grid):
        sig = base_sig(method, p)
        base = base_cache.get(sig)
        if base is None:
            base = BASE_FUNCS[method](img, p)
            base_cache[sig] = base
        d = dice(FIN_FUNCS[method](base, p), gt)
        out.append((pidx, d))
    return (method, out)


def _eval_worker(task):
    method, params, img_fn, msk_fn, g = task
    d = dice(seg_apply(method, read_img(img_fn), params), read_mask(msk_fn))
    return {"img": img_fn, "geom": g, "method": method, "dice": d}


def _pool():
    from multiprocessing import Pool, cpu_count
    return Pool(max(1, cpu_count() - 1))


# ----------------------------------------------------------------------------
# Tune (on val) -> freeze. Parallel over (method, image); each task sweeps the
# whole grid for that method using cached bases.
# ----------------------------------------------------------------------------
def tune_all(val_pairs):
    tasks = [(m, i, msk) for m in METHODS for i, msk in val_pairs]
    print(f"  grid sizes: " + ", ".join(f"{m}={len(PARAM_GRID[m])}" for m in METHODS))
    sums = {m: np.zeros(len(PARAM_GRID[m])) for m in METHODS}
    counts = {m: 0 for m in METHODS}
    with _pool() as pool:
        for method, out in pool.imap_unordered(_tune_worker, tasks, chunksize=1):
            for pidx, d in out:
                sums[method][pidx] += d
            counts[method] += 1
    frozen = {}
    for m in METHODS:
        means = sums[m] / max(counts[m], 1)
        best_pi = int(np.argmax(means))
        frozen[m] = PARAM_GRID[m][best_pi]
        print(f"  [{m}] best val Dice={means[best_pi]:.4f} "
              f"(of {len(means)} configs)  params={frozen[m]}")
    return frozen


# ----------------------------------------------------------------------------
# Evaluate (on test) with frozen params. Parallel over (method, image).
# ----------------------------------------------------------------------------
def evaluate_all(frozen, test_pairs, geom):
    tasks = [(m, frozen[m], i, msk, geom.get(i, "??"))
             for m in METHODS for i, msk in test_pairs]
    rows = []
    with _pool() as pool:
        for n, r in enumerate(pool.imap_unordered(_eval_worker, tasks, chunksize=4), 1):
            rows.append(r)
            if n % 500 == 0:
                print(f"    eval {n}/{len(tasks)}...", flush=True)
    return rows


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def cmd_download(args):
    val = stratified_sample("val", args.tune_per_geom)
    test = stratified_sample("test", args.test_per_geom)
    print("Downloading val (tuning) set...")
    s3_download(val)
    print("Downloading test (eval) set...")
    s3_download(test)
    json.dump({"val": val, "test": test},
              open(os.path.join(OUT_DIR, "sampled_pairs.json"), "w"))
    print(f"Done. val={len(val)} test={len(test)} cached.")


def cmd_run(args):
    geom = load_geom()
    pairs = json.load(open(os.path.join(OUT_DIR, "sampled_pairs.json")))
    val, test = [tuple(x) for x in pairs["val"]], [tuple(x) for x in pairs["test"]]

    frozen_path = os.path.join(OUT_DIR, "frozen_params.json")
    if os.path.exists(frozen_path):
        print("== Loading cached frozen params (skip retune) ==")
        raw = json.load(open(frozen_path))
        frozen = {}
        for m, p in raw.items():
            p = dict(p)
            if "sigmas" in p:
                p["sigmas"] = np.array(p["sigmas"])
            frozen[m] = p
    else:
        print("== TUNE on val, then freeze ==")
        frozen = tune_all(val)
        json.dump({m: {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in p.items()} for m, p in frozen.items()},
                  open(frozen_path, "w"), indent=2)

    print("== EVAL on frozen test ==")
    all_rows = evaluate_all(frozen, test, geom)

    with open(os.path.join(OUT_DIR, "per_image_dice.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["img", "geom", "method", "dice"])
        w.writeheader()
        w.writerows(all_rows)

    # Summary table: method x geometry mean Dice
    print("\n=== Mean Dice (method x geometry) ===")
    hdr = f"{'method':<10}" + "".join(f"{g:>10}" for g in GEOMS) + f"{'ALL':>10}"
    print(hdr)
    summary = {}
    for m in METHODS:
        line = f"{m:<10}"
        per = {}
        for g in GEOMS:
            vals = [r["dice"] for r in all_rows if r["method"] == m and r["geom"] == g]
            per[g] = float(np.mean(vals)) if vals else float("nan")
            line += f"{per[g]:>10.4f}"
        allv = [r["dice"] for r in all_rows if r["method"] == m]
        per["ALL"] = float(np.mean(allv))
        line += f"{per['ALL']:>10.4f}"
        summary[m] = per
        print(line)
    json.dump(summary, open(os.path.join(OUT_DIR, "dice_summary.json"), "w"), indent=2)
    print(f"\nWrote out/per_image_dice.csv, out/dice_summary.json, out/frozen_params.json")
    print("U-Net per-geometry test Dice (test_dice_Structure_*) to be added from the "
          "R4-full MLflow run once it lands.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="stratified download of val(tune)+test(eval)")
    d.add_argument("--tune-per-geom", type=int, default=30,
                   help="images per geometry for tuning (val)")
    d.add_argument("--test-per-geom", type=int, default=60,
                   help="images per geometry for eval (test); use large for full")
    d.set_defaults(func=cmd_download)

    r = sub.add_parser("run", help="tune-freeze-eval; emit Dice table")
    r.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
