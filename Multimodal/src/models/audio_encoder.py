import torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self, audio_dropout: float, embedding_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(audio_dropout),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(audio_dropout),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.proj = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.Dropout(audio_dropout)
        )
        
        
    def forward(self, x):
        x = self.encoder(x) # [B, 128, 1, 1]
        x = x.flatten(1) # [B, 128]
        x = self.proj(x) # [B, embedding_dim]
        return x