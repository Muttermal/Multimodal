# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/20 15:03
# @Author  : zhangguangyi
# @File    : __init__.py.py

from .image_models import ImageModel
from .text_models import TextModel

from .clip import CLIPModel
from .component import pooling
from .ViTAE_Window_NoShift.models import ViTAEWindowNoShiftBasic
from .swin_transformer import SwinTransformer
