import logging
import math
import os
from collections import Counter
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class DistributedWeightedSampler(Sampler):
    """Weighted random sampler that is safe to use under DDP.

    Each replica draws ``num_samples`` indices per epoch by weighted sampling (with
    replacement) from the full dataset using the provided per-sample weights. A single
    global pool is drawn with an epoch-seeded generator and then strided by ``rank`` so
    that replicas stay disjoint within an epoch while every rank uses the same weighting.

    This is used to equalize per-epoch class (geometry) frequency without discarding
    data: under-represented classes are revisited (oversampled) and over-represented
    classes are drawn less often, but every sample remains eligible. When
    ``num_replicas == 1`` this behaves like a standard ``WeightedRandomSampler`` whose
    epoch can be advanced for reproducible shuffling.

    NOTE: When this sampler is attached to the training DataLoader, the PyTorch Lightning
    ``Trainer`` must be created with ``use_distributed_sampler=False`` so Lightning does
    not replace it with a plain ``DistributedSampler``.
    """

    def __init__(
        self,
        weights: Sequence[float],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        replacement: bool = True,
        seed: int = 42,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0

        self.total_size = len(self.weights)
        # Per-replica sample count so the union across replicas ~= one pass over the data
        self.num_samples = math.ceil(self.total_size / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        """Advance the epoch so each epoch draws a different (but reproducible) sample.

        PyTorch Lightning calls this automatically on samplers that define it.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        # Draw one global pool for all replicas, then take this rank's strided shard.
        total_draw = self.num_samples * self.num_replicas
        pool = torch.multinomial(self.weights, total_draw, self.replacement, generator=g)
        indices = pool[self.rank : total_draw : self.num_replicas]
        return iter(indices.tolist())


def build_class_balanced_weights(
    image_paths: Sequence[str],
    label_map: dict,
    balance_column: str = "Structure",
) -> list:
    """Compute per-sample weights that equalize class frequency across the training set.

    Each sample's weight is ``1 / count(class)`` so that, in expectation, every class is
    drawn equally often per epoch regardless of how many images it contributes.

    :param image_paths: Ordered training image paths (order must match the dataset).
    :param label_map: Mapping of image basename -> class label (e.g. "FCT"/"SC"/"HELI").
    :param balance_column: Name of the class field, used only for logging.
    :return: List of float weights aligned 1:1 with ``image_paths``.
    """
    labels = [label_map.get(os.path.basename(p).strip()) for p in image_paths]

    missing = sum(1 for label in labels if label is None)
    if missing:
        logger.warning(
            f"Class-balanced sampler: {missing}/{len(labels)} training images have no "
            f"'{balance_column}' label and will be assigned the mean class weight."
        )

    counts = Counter(label for label in labels if label is not None)
    if not counts:
        raise ValueError(
            f"Class-balanced sampler enabled but no '{balance_column}' labels matched the "
            f"training images. Check that the metadata file covers the training split."
        )

    logger.info(f"Class-balanced sampler: per-epoch frequency equalized across {dict(counts)}")

    class_weight = {label: 1.0 / count for label, count in counts.items()}
    mean_weight = sum(class_weight.values()) / len(class_weight)
    return [class_weight.get(label, mean_weight) for label in labels]
