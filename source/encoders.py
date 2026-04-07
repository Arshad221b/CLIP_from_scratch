import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel, DistilBertTokenizer


class ImageEncoder(nn.Module):
    """Lightweight CNN for MacBook training."""

    def __init__(self, embed_dim=256):
        super().__init__()
        # Use MobileNetV3-Small - very efficient
        backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Project to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(576, 512), nn.ReLU(), nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        x = self.projection(x)
        return x


class TextEncoder(nn.Module):
    """DistilBERT-based text encoder (smaller than BERT)."""

    def __init__(self, embed_dim=256):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.projection = nn.Linear(768, embed_dim)

    def forward(self, input_ids, attention_mask):
        # Get CLS token representation
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.projection(cls_token)


class CLIPModel(nn.Module):
    """Complete CLIP model combining image and text encoders."""

    def __init__(self, embed_dim=256, temperature=0.07):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)
        self.temperature = nn.Parameter(torch.ones([]) * temperature)

    def forward(self, images, input_ids, attention_mask):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        return image_features, text_features
