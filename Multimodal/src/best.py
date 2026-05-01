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
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0.0001
    use_weighted_loss: bool = True
    use_weighted_sampler: bool = True 
    label_smoothing: float = 0.2
    alpha: float =  0.5
    beta: float = 0.2
    early_stopping_patience: int = 6
    modality_dropout_prob: float = 0.1

    # Visual encoder
    visual_encoder_lr: float = 5.802871660923551e-06
    train_last_n_blocks_visual: int = 0
    visual_encoder_proj_lr: float = 1e-4

    # Audio encoder
    audio_encoder_lr = 7.485781293933215e-06

    # Text encoder
    text_encoder_lr: float = 1.1349557175786905e-06
    text_encoder_proj_lr: float = 1e-4
    train_last_n_blocks_text: int = 1

    # Gating
    gate_lr: float =  1.2116982048494386e-05

    # Final classifier
    classifier_lr: float = 2.058953686247326e-05

    # Contrastive loss
    use_contrastive: bool = True
    contrastive_weight: float = 0.15
    contrastive_temperature: float = 0.1


@dataclass
class ModelConfig:
    modalities: List[str] = field(default_factory=lambda: ["image", "text", "audio"])
    embedding_dim: int = 256
    num_classes: int = 7
    classifier_dropout: float = 0.5

    gated: bool = True
    visual_encoder: str = "inception"
    visual_dropout: float = 0.3

    text_encoder: str = "distilbert-base-uncased"
    text_dropout: float = 0.4
    
    audio_encoder: str = "cnn"
    audio_dropout: float = 0.3

@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output_dir: str = Path(ROOT/"outputs")