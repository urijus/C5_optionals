import torch
from pathlib import Path
from src.train.losses import build_loss
from src.data.sampler import compute_sample_weights



def move_batch_to_device(batch, device):
    moved = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value

    return moved


def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


# Set different LR for different parts of the model
def create_param_groups(config, model):
    param_groups = []

    # Visual encoder (if exists)
    if hasattr(model, "visual_encoder") and model.visual_encoder is not None:
        if hasattr(model.visual_encoder, "backbone"):
            param_groups.append({
                "params": [p for p in model.visual_encoder.backbone.parameters() if p.requires_grad],
                "lr": config.train.visual_encoder_lr,
            })

        if hasattr(model.visual_encoder, "proj"):
            param_groups.append({
                "params": [p for p in model.visual_encoder.proj.parameters() if p.requires_grad],
                "lr": config.train.visual_encoder_proj_lr,
            })

    # Always add classifier
    param_groups.append({
        "params": [p for p in model.classifier.parameters() if p.requires_grad],
        "lr": config.train.classifier_lr,
    })

    return param_groups

def create_param_groups(config, model):
    param_groups = []

    # Visual encoder
    if hasattr(model, "visual_encoder") and model.visual_encoder is not None:
        if hasattr(model.visual_encoder, "backbone"):
            params = [p for p in model.visual_encoder.backbone.parameters() if p.requires_grad]
            if len(params) > 0:
                param_groups.append({
                    "params": params,
                    "lr": config.train.visual_encoder_lr,
                })

        if hasattr(model.visual_encoder, "proj"):
            params = [p for p in model.visual_encoder.proj.parameters() if p.requires_grad]
            if len(params) > 0:
                param_groups.append({
                    "params": params,
                    "lr": config.train.visual_encoder_proj_lr,
                })

    # Audio encoder
    if hasattr(model, "audio_encoder") and model.audio_encoder is not None:
        params = [p for p in model.audio_encoder.parameters() if p.requires_grad]
        if len(params) > 0:
            param_groups.append({
                "params": params,
                "lr": config.train.audio_encoder_lr,
            })

    # Text encoder
    if hasattr(model, "text_encoder") and model.text_encoder is not None:
        if hasattr(model.text_encoder, "backbone"):
            params = [p for p in model.text_encoder.backbone.parameters() if p.requires_grad]
            if len(params) > 0:
                param_groups.append({
                    "params": params,
                    "lr": config.train.text_encoder_lr,
                })

        if hasattr(model.text_encoder, "proj"):
            params = [p for p in model.text_encoder.proj.parameters() if p.requires_grad]
            if len(params) > 0:
                param_groups.append({
                    "params": params,
                    "lr": config.train.text_encoder_proj_lr,
                })

    # Classifier
    params = [p for p in model.classifier.parameters() if p.requires_grad]
    if len(params) > 0:
        param_groups.append({
            "params": params,
            "lr": config.train.classifier_lr,
        })

    return param_groups


def save_checkpoint(checkpoint, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)


def load_checkpoint(model, checkpoint_path, device, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def train_one_epoch(
        model, 
        train_loader, 
        optimizer, 
        class_weights,
        ethnic_weights,
        alpha,
        beta,
        loss_fn, 
        device):
    
    model.train()

    running_loss = 0.0
    running_acc = 0.0

    class_weights = class_weights.to(device) if class_weights is not None else None
    ethnic_weights = ethnic_weights.to(device) if ethnic_weights is not None else None

    for batch in train_loader:
        batch = move_batch_to_device(batch, device)
        labels = batch["age"]
        ethnicities = batch["ethnicity"]

        optimizer.zero_grad()

        logits = model(batch)
        loss_per_sample = loss_fn(logits, labels)


        if class_weights is not None and ethnic_weights is not None:
            sample_weights = compute_sample_weights(
                labels=labels,
                ethnicities=ethnicities,
                class_weights=class_weights,
                ethnic_weights=ethnic_weights,
                alpha=alpha,
                beta=beta,
            )
            loss = (loss_per_sample * sample_weights).mean()
        else:
            loss = loss_per_sample.mean()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += compute_accuracy(logits, labels)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_acc / len(train_loader)

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, data_loader, loss_fn, device):
    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        labels = batch["age"]

        logits = model(batch)
        loss = loss_fn(logits, labels).mean()

        running_loss += loss.item()
        running_acc += compute_accuracy(logits, labels)

    epoch_loss = running_loss / len(data_loader)
    epoch_acc = running_acc / len(data_loader)

    return epoch_loss, epoch_acc


def fit(config, model, train_loader, val_loader, class_weights, ethnic_weights, device):
    param_groups = create_param_groups(config, model)

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.train.weight_decay,
    )

    loss_fn = build_loss(label_smoothing=config.train.label_smoothing)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    config.output_dir.mkdir(exist_ok=True, parents=True)
    for epoch in range(config.train.epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            class_weights=class_weights,
            ethnic_weights=ethnic_weights,
            alpha=config.train.alpha,
            beta=config.train.beta,
            device=device
        )

        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1:03d}/{config.train.epochs:03d} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }

            save_checkpoint(best_state, config.output_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.train.early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return history, best_state