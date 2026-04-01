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
    max_text_length: int = 128
    num_workers: int = 4

@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    use_weighted_loss: bool = True
    use_weighted_sampler: bool = True
    early_stopping_patience: int = 10

@dataclass
class ModelConfig:
    modalities: Optional[List[str]] = None
    embedding_dim: int = 256
    num_classes: int = 7
    dropout: float = 0.3
    visual_encoder: str = "convnextv2"

@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output_dir: str = Path(ROOT/"outputs").mkdir(exist_ok=True)