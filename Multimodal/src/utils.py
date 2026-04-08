import torch
from src.train.engine import move_batch_to_device


@torch.no_grad()
def inspect_average_gates(model, data_loader, device):
    model.eval()

    gate_sums = {m: 0.0 for m in model.modalities}
    num_batches = 0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        _, gate_dict = model(batch, return_gates=True)

        for modality, gate in gate_dict.items():
            gate_sums[modality] += gate.mean().item()

        num_batches += 1

    avg_gates = {m: gate_sums[m] / num_batches for m in gate_sums}
    return avg_gates