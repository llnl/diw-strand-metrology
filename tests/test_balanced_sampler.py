import os
from collections import Counter

import pytest

from llnl_ml.data.balanced_sampler import (
    DistributedWeightedSampler,
    build_class_balanced_weights,
)


def _make_corpus():
    """Imbalanced corpus mirroring the DIW geometry split (FCT >> HELI > SC)."""
    labels = ["FCT"] * 802 + ["HELI"] * 424 + ["SC"] * 153
    paths = [f"/img/{i:05d}_{lab}-unproc.png" for i, lab in enumerate(labels)]
    label_map = {os.path.basename(p): p.split("_")[-1].replace("-unproc.png", "") for p in paths}
    return paths, label_map


def test_weights_inverse_to_class_frequency():
    paths, label_map = _make_corpus()
    weights = build_class_balanced_weights(paths, label_map)

    assert len(weights) == len(paths)
    # Rarer classes get larger weights: SC (153) > HELI (424) > FCT (802)
    w_by_class = {label_map[os.path.basename(p)]: w for p, w in zip(paths, weights)}
    assert w_by_class["SC"] > w_by_class["HELI"] > w_by_class["FCT"]


def test_missing_labels_get_mean_weight():
    paths, label_map = _make_corpus()
    paths.append("/img/99999_unlabeled-unproc.png")  # not in label_map
    weights = build_class_balanced_weights(paths, label_map)
    assert len(weights) == len(paths)
    assert weights[-1] > 0


def test_no_matching_labels_raises():
    paths, _ = _make_corpus()
    with pytest.raises(ValueError):
        build_class_balanced_weights(paths, {"nope.png": "SC"})


def test_single_replica_equalizes_class_frequency():
    paths, label_map = _make_corpus()
    weights = build_class_balanced_weights(paths, label_map)

    sampler = DistributedWeightedSampler(weights, num_replicas=1, rank=0, seed=42)
    sampler.set_epoch(0)
    drawn = list(sampler)

    geom = Counter(label_map[os.path.basename(paths[i])] for i in drawn)
    total = sum(geom.values())
    fracs = [count / total for count in geom.values()]
    # Raw fractions are ~0.58/0.31/0.11; balanced should be ~1/3 each.
    assert max(fracs) - min(fracs) < 0.06


def test_epoch_reshuffles_reproducibly():
    paths, label_map = _make_corpus()
    weights = build_class_balanced_weights(paths, label_map)
    sampler = DistributedWeightedSampler(weights, num_replicas=1, rank=0, seed=42)

    sampler.set_epoch(0)
    e0 = list(sampler)
    sampler.set_epoch(1)
    e1 = list(sampler)
    sampler.set_epoch(0)
    e0_again = list(sampler)

    assert e0 != e1  # different epochs draw differently
    assert e0 == e0_again  # same epoch is reproducible


def test_ddp_shards_are_equal_length_and_cover_epoch():
    paths, label_map = _make_corpus()
    weights = build_class_balanced_weights(paths, label_map)

    shards = []
    for rank in range(4):
        sampler = DistributedWeightedSampler(weights, num_replicas=4, rank=rank, seed=42)
        sampler.set_epoch(0)
        shards.append(list(sampler))

    lengths = [len(s) for s in shards]
    assert all(length == lengths[0] for length in lengths)
    assert sum(lengths) >= len(paths)
