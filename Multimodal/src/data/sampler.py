import torch
from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler(labels):
    """
    Builds a WeightedRandomSampler using inverse class frequency.

    Args:
        labels: list or 1D tensor of integer class labels (0..num_classes-1)

    Returns:
        sampler: WeightedRandomSampler
        class_counts: tensor with counts per class
        class_weights: tensor with inverse-frequency weights per class
    """
    labels = torch.as_tensor(labels, dtype=torch.long)

    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()

    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler, class_counts, class_weights