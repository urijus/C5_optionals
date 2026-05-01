import torch
import torch.nn as nn
from itertools import combinations
import torch.nn.functional as F


def build_loss(label_smoothing=0.0):
    return nn.CrossEntropyLoss(
        reduction="none",
        label_smoothing=label_smoothing,
    )


def pairwise_contrastive_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)

    loss_1 = F.cross_entropy(logits, labels)
    loss_2 = F.cross_entropy(logits.T, labels)

    return 0.5 * (loss_1 + loss_2)


def multimodal_contrastive_loss(feat_dict, temperature=0.1):
    modalities = list(feat_dict.keys())

    if len(modalities) < 2:
        return torch.tensor(0.0, device=next(iter(feat_dict.values())).device)

    losses = []
    for m1, m2 in combinations(modalities, 2):
        losses.append(pairwise_contrastive_loss(feat_dict[m1], feat_dict[m2], temperature))

    return sum(losses) / len(losses)