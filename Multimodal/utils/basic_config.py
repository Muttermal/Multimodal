# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/5 10:00
# @Author  : zhangguangyi
# @File    : basic_config.py
import json


class BasicConfig:

    model_type: str = "Base"

    def __init__(self, **kwargs):
        super(BasicConfig, self).__init__()

    def __repr__(self):
        return f"config of {self.model_type} model"

    @classmethod
    def load_from_json(cls, json_file):
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)
