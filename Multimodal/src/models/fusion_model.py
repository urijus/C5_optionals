import random
import torch
import torch.nn as nn
from src.models.visual_encoder import VisualEncoder
from src.models.audio_encoder import AudioEncoder
from src.models.text_encoder import TextEncoder


 
class MultiModalModel(nn.Module):
    def __init__(self, config):
            super().__init__()
            self.modalities = config.model.modalities
            self.embedding_dim = config.model.embedding_dim
            self.num_classes = config.model.num_classes
            self.modality_dropout_prob = config.train.modality_dropout_prob

            if "image" in self.modalities:
                visual_model = config.model.visual_encoder
                train_last_n_blocks = config.train.train_last_n_blocks_visual
                self.visual_encoder = VisualEncoder(
                    visual_model, 
                    self.embedding_dim,
                    train_last_n_blocks)
            if "audio" in self.modalities:
                self.audio_encoder = AudioEncoder(self.embedding_dim)
            if "text" in self.modalities:
                text_model = config.model.text_encoder
                train_last_n_blocks = config.train.train_last_n_blocks_text
                self.text_encoder = TextEncoder(
                    self.embedding_dim, 
                    text_model, 
                    train_last_n_blocks)

            fusion_in_dim = self.embedding_dim * len(self.modalities)
            self.classifier = nn.Sequential(
                nn.Linear(fusion_in_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.num_classes),
            )

    def forward(self, batch):
        feats = []

        if "image" in self.modalities:
            feats.append(self.visual_encoder(batch["image"]))
        if "audio" in self.modalities:
            feats.append(self.audio_encoder(batch["audio"]))
        if "text" in self.modalities:
            feats.append(
                self.text_encoder(batch["input_ids"], batch["attention_mask"]))

        feats = self.apply_modality_dropout(feats)
        fused = torch.cat(feats, dim=1) # could we use transformer rather than just concat?
        logits = self.classifier(fused)
        return logits
    
    def apply_modality_dropout(self, feats):
        if not self.training or self.modality_dropout_prob <= 0 or len(feats) <= 1:
            return feats

        keep_mask = []
        for _ in feats:
            keep = torch.rand(1).item() >= self.modality_dropout_prob
            keep_mask.append(keep)

        if not any(keep_mask):
            keep_mask[random.randrange(len(keep_mask))] = True

        out = []
        for feat, keep in zip(feats, keep_mask):
            out.append(feat if keep else torch.zeros_like(feat))
        return out



        # WHAT TO DO: 
        # 1) SAVE BY FAIRNESS
        # 2) ADD REGULARIZATION (HEAVY OVERFITTING)
        # 3) ADD MODALITY DROPUT (AUDIO DOESN NTO PLAUY A BIG ROLE NOW...)