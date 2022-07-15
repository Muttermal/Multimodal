# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/21 13:49
# @Author  : zhangguangyi
# @File    : data_augmentation.py
"""数据增强"""
import albumentations as A
from typing import List


def get_transforms(args, mode="train"):
    """图像数据增强"""

    if mode == "train":
        transforms = A.Compose(
            [A.Resize(args.size, args.size, always_apply=True), A.Normalize(max_pixel_value=255.0, always_apply=True), ]
        )
    else:
        transforms = A.Compose(
            [A.Resize(args.size, args.size, always_apply=True), A.Normalize(max_pixel_value=255.0, always_apply=True), ]
        )
    return transforms


def text_aug(args, mode: List[str, ...]):
    pass
