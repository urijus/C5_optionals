import torch


def build_multimodal_collate_fn(tokenizer=None, max_text_length=260):
    def multimodal_collate_fn(batch):
        out = {}

        out["id"] = [sample["id"] for sample in batch]
        out["user_id"] = [sample["user_id"] for sample in batch]
        out["age"] = torch.tensor([sample["age"] for sample in batch], dtype=torch.long)
        out["gender"] = torch.tensor([sample["gender"] for sample in batch], dtype=torch.long)
        out["ethnicity"] = torch.tensor([sample["ethnicity"] for sample in batch], dtype=torch.long)

        out["image"] = (
            torch.stack([sample["image"] for sample in batch], dim=0)
            if batch[0]["image"] is not None else None
        )

        out["audio"] = (
            torch.stack([sample["audio"] for sample in batch], dim=0)
            if batch[0]["audio"] is not None else None
        )

        if batch[0]["text"] is not None:
            texts = [sample["text"] for sample in batch]
            out["text"] = texts

            if tokenizer is not None:
                encoded = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_text_length,
                    return_tensors="pt",
                )
                out["input_ids"] = encoded["input_ids"]
                out["attention_mask"] = encoded["attention_mask"]
            else:
                out["input_ids"] = None
                out["attention_mask"] = None
        else:
            out["text"] = None
            out["input_ids"] = None
            out["attention_mask"] = None

        return out

    return multimodal_collate_fn