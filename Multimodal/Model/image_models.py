# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/22 10:16
# @Author  : zhangguangyi
# @File    : image_models.py

import timm
from timm.models import swin_transformer
import torch.nn as nn


class ImageModel(nn.Module):

    def __init__(self, args):
        super(ImageModel, self).__init__()
        self.model_name = args.image_model
        self.model_path = args.image_model_path
        self.model = self.get_model()

    def get_model(self):
        model = timm.create_model(model_name=self.model_name, pretrained=True, checkpoint_path=self.model_path)
        return model

    def forward(self, x):
        result = self.model(x)
        return result
