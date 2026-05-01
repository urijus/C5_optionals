import torch.nn as nn
from transformers import AutoModel


def set_trainable_backbone(backbone, train_last_n_blocks):
    """
    Controls how much of the backbone is trainable.

    train_last_n_blocks:
        -1 : train full backbone
         0 : freeze full backbone
        >0 : train last N logical blocks/stages
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

    if "dist" in str(backbone.__class__.__name__).lower():
        # DistilBERT
        for layer in backbone.transformer.layer[-train_last_n_blocks:]:
            for p in layer.parameters():
                p.requires_grad = True
    else:
        # Bert
        for layer in backbone.encoder.layer[-train_last_n_blocks:]:
            for p in layer.parameters():
                p.requires_grad = True
        

class TextEncoder(nn.Module):
    def __init__(self, 
                 text_dropout: float,
                 embedding_dim: int, 
                 model_name: str = "distilbert-base-uncased", 
                 train_last_n_blocks=-1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        set_trainable_backbone(self.backbone, train_last_n_blocks=train_last_n_blocks)
        hidden_size = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden_size, embedding_dim)
        self.dropout = nn.Dropout(text_dropout)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(outputs, "last_hidden_state"):
            cls_token = outputs.last_hidden_state[:, 0]   # [B, hidden]
        else:
            cls_token = outputs[0][:, 0]

        cls_token = self.dropout(cls_token)
        x = self.proj(cls_token)
        return x