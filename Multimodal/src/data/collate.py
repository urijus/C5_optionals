import torch


def multimodal_collate_fn(batch):
    """
    Collates a list of samples from MultiModalDataset into a single batch dictionary.
    """
    out = {}

    # Metadata
    out["id"] = [sample["id"] for sample in batch]
    out["user_id"] = [sample["user_id"] for sample in batch]

    # Labels info
    out["age"] = torch.tensor(
        [sample["age"] for sample in batch],
        dtype=torch.long
    )
    out["gender"] = torch.tensor(
        [sample["gender"] for sample in batch],
        dtype=torch.long
    )
    out["ethnicity"] = torch.tensor(
        [sample["ethnicity"] for sample in batch],
        dtype=torch.long
    )

    # Modalities
    if batch[0]["image"] is not None:
        out["image"] = torch.stack([sample["image"] for sample in batch], dim=0)
    else:
        out["image"] = None

    if batch[0]["audio"] is not None:
        out["audio"] = torch.stack([sample["audio"] for sample in batch], dim=0)
    else:
        out["audio"] = None

    if batch[0]["text"] is not None:
        out["text"] = [sample["text"] for sample in batch]
    else:
        out["text"] = None

    return out