# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/1 10:04
# @Author  : zhangguangyi
# @File    : ViTAE.py

from functools import partial
from timm.models.layers import trunc_normal_
import torch
import torch.nn as nn

from ..utils import BasicConfig


class ModelConfig(BasicConfig):
    model_type: str = "ViTAE"

    def __init__(self, **kwargs):
        super(ModelConfig, self).__init__()
        self.img_size = kwargs.pop("img_size", 224)

        # ReductionCell parameters
        self.kernel_size = kwargs.pop("kernel_size", 4)
        self.down_sample_ratio = kwargs.pop("down_sample_ratio", 4)
        self.dilation = kwargs.pop("dilation", [1, 6, 12])
        self.in_channel = kwargs.pop("in_channel", 3)
        self.embed_dim = kwargs.pop("embed_dim", 64)
        self.share_weights = kwargs.pop("share_weights", False)
        self.op = kwargs.pop("op", "cat")

        # NormCell parameters
        self.dim = kwargs.pop("dim")
        self.num_heads = kwargs.pop("num_heads")
        self.mlp_ratio = kwargs.pop("mlp_ratio", 4)
        self.qkv_bias = kwargs.pop("qkv_bias", False)
        self.qk_scale = kwargs.pop("qk_scale", None)
        self.drop = kwargs.pop("drop", 0.)
        self.attn_drop = kwargs.pop("attn_drop", 0.)
        self.drop_path = kwargs.pop("drop_path", 0.)
        self.act_layer = kwargs.pop("act_layer", nn.GELU)
        self.norm_layer = kwargs.pop("norm_layer", nn.LayerNorm)
        self.class_token = kwargs.pop("class_token", False)
        self.group = kwargs.pop("group", 64)
        self.tokens_type = kwargs.pop("tokens_type", "transformer")
        self.shift_size = kwargs.pop("shift_size", 0)
        self.window_size = kwargs.pop("window_size", 0)
        self.gamma = kwargs.pop("gamma", False)
        self.init_values = kwargs.pop("init_values", 1e-4)
        self.SE = kwargs.pop("SE", False)
        self.relative_pos = kwargs.pop("relative_pos", False)

        # PatchEmbedding parameters
        self.inter_channel = kwargs.pop("inter_channel", 32)
        self.out_channels = kwargs.pop("out_channels", 48)


class ReductionCell(nn.Module):
    """负责将多尺度上下文和本地信息嵌入到token中。"""

    def __init__(self, config: ModelConfig):
        super(ReductionCell, self).__init__()
        self.img_size = config.img_size
        self.dilation = config.dilation
        self.embed_dim = config.embed_dim
        self.down_sample_ratio = config.down_sample_ratio
        self.op = config.op
        self.kernel_size = config.kernel_size
        self.stride = config.down_sample_ratio
        self.share_weights = config.share_weights
        self.outSize = self.img_size // self.down_sample_ratio

    def forward(self, x):
        pass


class NormalCell(nn.Module):
    """用于进一步建模token中的局部性和长期依赖关系。"""

    def __init__(self, config: ModelConfig):
        super(NormalCell, self).__init__()

    def forward(self, x):
        pass


class PatchEmbedding(nn.Module):

    def __init__(self, config: ModelConfig):
        super(PatchEmbedding, self).__init__()

    def forward(self, x):
        pass


class BasicLayer(nn.Module):

    def __init__(self, rc_type: str):
        super(BasicLayer, self).__init__()
        self.rc_type = rc_type
        self.RC = self.get_rc_layer()
        self.NC = nn.ModuleList(NormalCell())

    def get_rc_layer(self):
        if self.rc_type == "stem":
            rc_layer = PatchEmbedding()
        elif self.rc_type == "normal":
            rc_layer = ReductionCell()
        else:
            rc_layer = nn.Identity()
        return rc_layer

    def forward(self, x):
        x = self.RC(x)
        for nc in self.NC:
            x = nc(x)
        return x


class ViTAE(nn.Module):
    """
    输入图像x,首先经过三个RC层进行下采样，得到的结果concatenate后加上余弦位置向量
    """

    def __init__(self, dropout_rate, stages):
        super(ViTAE, self).__init__()
        self.stage = stages
        self.pos_drop = nn.Dropout(dropout_rate)
        self.layers = []
        all_layers = [BasicLayer() for i in range(stages)]
        self.layers = nn.ModuleList(all_layers)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        embedding = torch.mean(x, 1)
        return embedding
