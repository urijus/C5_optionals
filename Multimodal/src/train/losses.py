import torch.nn as nn


def build_loss(label_smoothing=0.05):
    return nn.CrossEntropyLoss(
        reduction="none",
        label_smoothing=label_smoothing,
    )