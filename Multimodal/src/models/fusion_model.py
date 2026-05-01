import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.visual_encoder import VisualEncoder
from src.models.audio_encoder import AudioEncoder
from src.models.text_encoder import TextEncoder


class CrossModalGatedFusion(nn.Module):
    def __init__(self, embedding_dim: int, modalities: list[str]):
        super().__init__()
        self.modalities = modalities
        self.embedding_dim = embedding_dim
        self.num_modalities = len(modalities)

        total_dim = embedding_dim * self.num_modalities

        self.gates = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(total_dim, embedding_dim),
                nn.Sigmoid(),
            )
            for modality in modalities
        })

    def forward(self, feat_dict, return_gates=False):
        feats = [feat_dict[m] for m in self.modalities]
        concat_feats = torch.cat(feats, dim=1)   # [B, M*D]

        gated_feats = []
        gate_dict = {}

        for modality in self.modalities:
            gate = self.gates[modality](concat_feats)   # [B, D]
            gate_dict[modality] = gate
            gated_feats.append(feat_dict[modality] * gate)

        fused = torch.cat(gated_feats, dim=1)

        if return_gates:
            return fused, gate_dict
        return fused
 
class MultiModalModel(nn.Module):
    def __init__(self, config, device):
            super().__init__()
            self.modalities = config.model.modalities
            self.embedding_dim = config.model.embedding_dim
            self.num_classes = config.model.num_classes
            self.modality_dropout_prob = config.train.modality_dropout_prob
            self.classifier_dropout = config.model.classifier_dropout
            self.audio_dropout = config.model.audio_dropout
            self.text_dropout = config.model.text_dropout
            self.device = device
            self.gated = config.model.gated

            if "image" in self.modalities:
                visual_model = config.model.visual_encoder
                train_last_n_blocks = config.train.train_last_n_blocks_visual
                self.visual_encoder = VisualEncoder(
                    visual_model, 
                    self.embedding_dim,
                    train_last_n_blocks,
                    self.device
                    )
            if "audio" in self.modalities:
                self.audio_encoder = AudioEncoder(
                    self.audio_dropout,
                    self.embedding_dim)
            if "text" in self.modalities:
                text_model = config.model.text_encoder
                train_last_n_blocks = config.train.train_last_n_blocks_text
                self.text_encoder = TextEncoder(
                    self.text_dropout,
                    self.embedding_dim, 
                    text_model, 
                    train_last_n_blocks)
                
            self.gated_fusion = CrossModalGatedFusion(
                embedding_dim=self.embedding_dim,
                modalities=self.modalities,
            ) if config.model.gated else None

            fusion_in_dim = self.embedding_dim * len(self.modalities)
            self.classifier = nn.Sequential(
                nn.Linear(fusion_in_dim, 128),
                nn.ReLU(),
                nn.Dropout(self.classifier_dropout),
                nn.Linear(128, self.num_classes),
            )

    def forward(self, batch, return_gates=False, return_features=False):
        raw_feat_dict = {}

        if "image" in self.modalities:
            raw_feat_dict["image"] = F.normalize(self.visual_encoder(batch["image"]), dim=1)

        if "audio" in self.modalities:
            raw_feat_dict["audio"] = F.normalize(self.audio_encoder(batch["audio"]), dim=1)

        if "text" in self.modalities:
            raw_feat_dict["text"] = F.normalize(
                self.text_encoder(batch["input_ids"], batch["attention_mask"]), dim=1
            )

        feats = [raw_feat_dict[m] for m in self.modalities]
        dropped_feats = self.apply_modality_dropout(feats)

        feat_dict = {m: feat for m, feat in zip(self.modalities, dropped_feats)}

        if self.gated:
            if return_gates:
                fused, gate_dict = self.gated_fusion(feat_dict, return_gates=True)
                logits = self.classifier(fused)
                if return_features:
                    return logits, gate_dict, raw_feat_dict
                return logits, gate_dict

            fused = self.gated_fusion(feat_dict)
        else:
            fused = torch.cat(dropped_feats, dim=1)

        logits = self.classifier(fused)

        if return_features:
            return logits, raw_feat_dict

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