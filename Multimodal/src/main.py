import argparse
from src.config import Config
from src.data.loaders import get_dataloaders
from src.models.fusion_model import MultiModalModel


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




if __name__=="__main__":
    args = get_user_args()
    config = Config()

    if args.seed is not None:
        config.seed = args.seed
    if args.modalities is not None:
        config.model.modalities = args.modalities

    # Load data
    train_loader, valid_loader, test_loader, class_counts, class_weights = get_dataloaders(config)

    # Train
    model = MultiModalModel(config)