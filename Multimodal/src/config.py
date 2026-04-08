from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

@dataclass
class DataConfig:
    root_dir: str = ROOT
    data_dir: str = Path(ROOT/ "data")
    image_size: int = 224
    sample_rate: int = 8000
    n_mels: int = 64
    max_text_length: int = 256
    num_workers: int = 4

@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    use_weighted_loss: bool = True
    use_weighted_sampler: bool = True
    alpha: float = 0.5
    beta: float = 0.1
    label_smoothing: float = 0.05
    early_stopping_patience: int = 6
    modality_dropout_prob: float = 0.1

    # Visual encoder
    visual_encoder_lr: float = 1e-5
    train_last_n_blocks_visual: int = 2
    visual_encoder_proj_lr: float = 1e-4

    # Audio encoder
    audio_encoder_lr = 1e-4

    # Text encoder
    text_encoder_lr: float = 5e-6
    text_encoder_proj_lr: float = 1e-4
    train_last_n_blocks_text: int = 2

    # Gating
    gate_lr: float = 1e-4

    # Final classifier
    classifier_lr: float = 1e-4

    modality_dropout_prob: float = 0.0
    

@dataclass
class ModelConfig:
    modalities: List[str] = field(default_factory=lambda: ["image", "text", "audio"])
    embedding_dim: int = 256
    num_classes: int = 7
    dropout: float = 0.4

    gated: bool = False
    visual_encoder: str = "inception"
    text_encoder: str = "distilbert-base-uncased"
    audio_encoder: str = "cnn"

@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output_dir: str = Path(ROOT/"outputs")