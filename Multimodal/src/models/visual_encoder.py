import timm
import torch.nn as nn

def build_backbone(encoder):
    backbone_map = {
        'rny002': 'regnety_002',
        'vit_base': 'vit_base_patch16_224',
        'eva02': 'eva02_small_patch14_224.mim_in22k',
        'convnextv2': 'convnextv2_tiny.fcmae_ft_in22k_in1k',
    }

    if encoder not in backbone_map:
        raise NotImplementedError(f"Unsupported visual encoder: {encoder}")
    
    model_name = backbone_map[encoder]
    features = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0
    )

    feat_dim = features.num_features
    data_config = timm.data.resolve_model_data_config(features)

    return features, feat_dim, data_config


def set_trainable_backbone(backbone, train_last_n_blocks):
    """
    Controls how much of the backbone is trainable.

    train_last_n_blocks:
        -1 : train full backbone
         0 : freeze full backbone
        >0 : train last N logical blocks/stages

    ViT:
        blocks = transformer blocks (+ final norm)

    RegNet/CNN:
        blocks = [stem, s1, s2, s3, s4, head]
    """
    # Train full backbone
    if train_last_n_blocks == -1:
        for p in backbone.parameters():
            p.requires_grad = True
        return

    # Freeze full backbone
    for p in backbone.parameters():
        p.requires_grad = False

    if train_last_n_blocks == 0:
        return

    # ViT-style backbones
    if hasattr(backbone, 'blocks'):
        trainable_groups = list(backbone.blocks)

        if hasattr(backbone, 'norm'):
            trainable_groups.append(backbone.norm)

        n = min(train_last_n_blocks, len(trainable_groups))

        for module in trainable_groups[-n:]:
            for p in module.parameters():
                p.requires_grad = True
        return

    # RegNet / CNN-style backbones
    trainable_groups = []

    # Order: early -> late
    for name in ['stem', 's1', 's2', 's3', 's4', 'head']:
        if hasattr(backbone, name):
            trainable_groups.append(getattr(backbone, name))

    if len(trainable_groups) == 0:
        return

    n = min(train_last_n_blocks, len(trainable_groups))

    for module in trainable_groups[-n:]:
        for p in module.parameters():
            p.requires_grad = True


class VisualEncoder(nn.Module):

    def __init__(self, visual_model: str, embedding_dim: int, train_last_n_blocks: int = -1):
        super().__init__()
        self.backbone, self.feat_dim, self.data_config = build_backbone(visual_model)
        set_trainable_backbone(self.backbone, train_last_n_blocks=train_last_n_blocks)
        self.proj = nn.Linear(self.feat_dim, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        features = self.proj(features) 
        return features  # [B, embedding_dim]