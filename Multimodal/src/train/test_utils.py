import pandas as pd
import torch

from src.train.engine import move_batch_to_device


@torch.no_grad()
def predict_and_export_csv(model, data_loader, device, save_path):
    model.eval()

    rows = []

    for batch in data_loader:
        # move tensor fields
        moved_batch = move_batch_to_device(batch, device)
        logits = model(moved_batch)
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        labels = moved_batch["age"].cpu().tolist()

        ids = batch["id"]

        for sample_id, gt, pred in zip(ids, labels, preds):
            rows.append({
                "VideoName": f"{sample_id}.mp4",
                "ground_truth": gt + 1,
                "prediction": pred + 1,
            })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    return df


@torch.no_grad()
def evaluate_test(model, data_loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in data_loader:
        moved_batch = move_batch_to_device(batch, device)
        labels = moved_batch["age"]
        logits = model(moved_batch)
        loss = loss_fn(logits, labels).mean()

        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc