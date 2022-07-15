# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/20 15:07
# @Author  : zhangguangyi
# @File    : text_models.py

from transformers import AutoModel
import torch.nn as nn


class TextModel(nn.Module):

    def __init__(self, args):
        super(TextModel, self).__init__()
        self.model_name = args.text_model
        self.model_path = args.text_model_path
        self.model = self.get_model()

    def get_model(self):
        model = AutoModel.from_pretrained(pretrained_model_name_or_path=self.model_path)
        return model

    def forward(self, x):
        result = self.model(x)
        return result
