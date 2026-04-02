import torch
from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler(labels, ethnicities, alpha=0.5, beta=0.1):
    labels = torch.as_tensor(labels, dtype=torch.long)
    ethnicities = torch.as_tensor(ethnicities, dtype=torch.long)

    class_counts = torch.bincount(labels, minlength=7)
    ethnic_counts = torch.bincount(ethnicities, minlength=3)

    class_weights = torch.where(
        class_counts > 0,
        1.0 / class_counts.float(),
        0.0,
    )

    ethnic_weights = torch.where(
        ethnic_counts > 0,
        1.0 / ethnic_counts.float(),
        0.0,
    )

    sample_weights = (class_weights[labels] ** alpha) * (ethnic_weights[ethnicities] ** beta)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler, class_weights, ethnic_weights


def compute_sample_weights(labels, ethnicities, class_weights, ethnic_weights, alpha=0.5, beta=0.1):
    age_w = class_weights[labels]
    eth_w = ethnic_weights[ethnicities]

    sample_w = (age_w ** alpha) * (eth_w ** beta)
    sample_w = sample_w / sample_w.mean()

    return sample_w