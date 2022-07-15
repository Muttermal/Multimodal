# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/20 15:08
# @Author  : zhangguangyi
# @File    : clip.py

import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import AutoModel

from .component import pooling


class ImageEncoder(nn.Module):
    """Encode images to a fixed size vector"""

    def __init__(
        self, model_name: [str, nn.Module], model_path: str, pretrained=True, trainable=True
    ):
        super().__init__()
        if isinstance(model_name, str):
            self.model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                checkpoint_path=model_path,
                num_classes=0,
                global_pool="avg"
            )
        else:
            self.model = model_name
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        result = self.model(x)
        return result


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, model_path: str, trainable=True):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_path)
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, inputs):
        text_encoder = self.model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
        text_embedding = pooling(text_encoder, inputs['attention_mask'], "mean", True)    # 对句子向量做mean pooling
        return text_embedding


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIPModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.image_encoder = ImageEncoder(model_name=args.image_model_name,
                                          model_path=args.image_model_path,
                                          pretrained=True, trainable=True)
        self.text_encoder = TextEncoder(model_name=args.text_model_name,
                                        model_path=args.text_model_path,
                                        trainable=True)
        self.image_projection = ProjectionHead(embedding_dim=args.image_embedding,
                                               projection_dim=args.projection_dim,
                                               dropout=args.dropout)
        self.text_projection = ProjectionHead(embedding_dim=args.text_embedding,
                                              projection_dim=args.projection_dim,
                                              dropout=args.dropout)
        self.temperature = args.temperature

    def __repr__(self):
        return "CLIP Model"

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch)
        text_features = self.text_encoder(batch)

        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = F.cross_entropy(logits, targets, reduction='none')
        images_loss = F.cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()
