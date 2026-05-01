import copy
import json
import shutil
from pathlib import Path

import torch
import optuna

from src.config import Config
from src.data.loaders import get_dataloaders
from src.models.fusion_model import MultiModalModel
from src.train.engine import fit, load_checkpoint
from src.train.losses import build_loss
from src.train.test_utils import evaluate_test


def set_nested_attr(obj, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def cleanup_run_dir(run_dir: Path):
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)


def run_experiment(config: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader, class_weights, ethnic_weights = get_dataloaders(config)

    model = MultiModalModel(config, device).to(device)

    history, best_state = fit(
        config,
        model,
        train_loader,
        valid_loader,
        class_weights,
        ethnic_weights,
        device,
    )

    # Load best checkpoint
    checkpoint_path = Path(config.output_dir) / "best_model.pt"
    load_checkpoint(model, checkpoint_path, device=device)

    # Evaluate on test
    loss_fn = build_loss()

    test_loss, test_acc, test_f1 = evaluate_test(model, test_loader, loss_fn, device)
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f} | Test macro-F1: {test_f1:.4f}")

    score = 0.6*test_acc + 0.4*test_f1

    return score #float(max(history["val_acc"]))

def objective(trial):
    cfg = copy.deepcopy(BASE_CONFIG)

    cfg.train.weight_decay = trial.suggest_categorical("train.weight_decay", [0.0, 1e-5, 1e-4, 7e-4])
    cfg.train.label_smoothing = trial.suggest_categorical("train.label_smoothing", [0.05, 0.1, 0.2])

    cfg.model.classifier_dropout = trial.suggest_categorical("model.classifier_dropout", [0.3, 0.4, 0.5])
    cfg.model.audio_dropout = trial.suggest_categorical("model.audio_dropout", [0.3, 0.4, 0.5])
    cfg.model.text_dropout = trial.suggest_categorical("model.text_dropout", [0.3, 0.4, 0.5])

    cfg.train.modality_dropout_prob = trial.suggest_categorical(
        "train.modality_dropout_prob", [0.1, 0.2, 0.3]
    )

    cfg.train.alpha = trial.suggest_float("train.alpha", 0.3, 0.6, log=False)
    cfg.train.beta = trial.suggest_float("train.beta", 0.2, 0.5, log=False)

    cfg.train.visual_encoder_lr = trial.suggest_float("train.visual_encoder_lr", 1e-6, 1e-5, log=True)
    cfg.train.audio_encoder_lr = trial.suggest_float("train.audio_encoder_lr", 1e-6, 1e-5, log=True)
    cfg.train.text_encoder_lr = trial.suggest_float("train.text_encoder_lr", 1e-6, 1e-5, log=True)
    cfg.train.gate_lr = trial.suggest_float("train.gate_lr", 1e-5, 5e-4, log=True)
    cfg.train.classifier_lr = trial.suggest_float("train.classifier_lr", 1e-5, 5e-4, log=True)

    cfg.train.train_last_n_blocks_visual = trial.suggest_categorical(
        "train.train_last_n_blocks_visual", [1, 2, 0, -1]
    )
    cfg.train.train_last_n_blocks_text = trial.suggest_categorical(
        "train.train_last_n_blocks_text", [1, 2, 0, -1]
    )

    cfg.train.contrastive_weight = trial.suggest_categorical("train.contrastive_weight", [0.1, 0.2, 0.05])

    run_dir = RESULTS_DIR / f"trial_{trial.number:03d}"
    cfg.output_dir = run_dir

    try:
        score = run_experiment(cfg)

        trial_result = {
            "trial": trial.number,
            "score": score,
            "params": trial.params,
        }
        with open(RESULTS_DIR / f"trial_{trial.number:03d}.json", "w") as f:
            json.dump(trial_result, f, indent=2)

        return score

    finally:
        cleanup_run_dir(run_dir)

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    # Save all trials
    trials_summary = []
    for t in study.trials:
        trials_summary.append({
            "trial": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        })

    with open(RESULTS_DIR / "optuna_trials.json", "w") as f:
        json.dump(trials_summary, f, indent=2)

    best_result = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial": study.best_trial.number,
    }

    with open(RESULTS_DIR / "optuna_best.json", "w") as f:
        json.dump(best_result, f, indent=2)

    print("\n" + "=" * 60)
    print("BEST RESULT")
    print("=" * 60)
    print(f"Best score: {study.best_value:.6f}")
    print(f"Best trial: {study.best_trial.number}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    print(f"Saved results to: {RESULTS_DIR}")


if __name__ == "__main__":
    BASE_CONFIG = Config()

    RESULTS_DIR = Path("optuna_results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    main()