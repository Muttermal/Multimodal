# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/13 9:58
# @Author  : zhangguangyi
# @File    : __init__.py
from pathlib import Path

USER_DIR = Path.expanduser(Path('~')).joinpath('.Multimodal')
if not USER_DIR.exists():
    USER_DIR.mkdir()

__version__ = "0.0.1.dev"
from . import DataLoader
from . import Loss
from . import Model
from . import utils
from . import trainer
