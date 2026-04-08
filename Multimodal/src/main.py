import argparse
import torch
from pathlib import Path
from src.config import Config
from src.data.loaders import get_dataloaders
from src.models.fusion_model import MultiModalModel
from src.train.engine import fit, load_checkpoint
from src.train.losses import build_loss
from src.train.test_utils import evaluate_test, predict_and_export_csv
from utils import inspect_average_gates


def get_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '-m', '--modalities',
        type=str,
        nargs='+',
        choices=["image", "audio", "text"],
        default=["image"],
        help="Modalities for the model to use."
    )
    return parser.parse_args()


def main():
    args = get_user_args()
    config = Config()

    if args.seed is not None:
        config.seed = args.seed
    if args.modalities is not None:
        config.model.modalities = args.modalities

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader, class_weights, ethnic_weights = get_dataloaders(config)

    model = MultiModalModel(config, device).to(device)

    # Train
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

    if config.model.gated:
        avg_gates = inspect_average_gates(model, valid_loader, device)

        print("\nAverage modality gates:")
        for modality, value in avg_gates.items():
            print(f"{modality}: {value:.4f}")

    # Evaluate on test
    loss_fn = build_loss(label_smoothing=0.0)

    test_loss, test_acc = evaluate_test(model, test_loader, loss_fn, device)
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

    # Export csv for bias evaluation script
    predict_and_export_csv(
        model=model,
        data_loader=test_loader,
        device=device,
        save_path=Path(config.output_dir) / "predictions_test_set.csv",
    )

if __name__=="__main__":
    main()
   

    