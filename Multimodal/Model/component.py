# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/23 16:59
# @Author  : zhangguangyi
# @File    : component.py

import torch


def pooling(hidden_tensor, attention_mask, type_, has_cls: True):
    """获取embedding vector经过不同pooling方式后的结果"""

    assert type_ in {"mean", "max", "cls", "mean_sqrt_len_tokens"}
    if has_cls:
        token_embeddings, cls_token_embeddings = hidden_tensor[0], hidden_tensor[1]
    else:
        token_embeddings, cls_token_embeddings = hidden_tensor, None

    if type_ == "cls":
        out_tensor = cls_token_embeddings
    elif type_ == "max":
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9
        out_tensor = torch.max(token_embeddings, 1)[0]
    elif type_ in {"mean", "mean_sqrt_len_tokens"}:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        if type_ == "mean":
            out_tensor = sum_embeddings / sum_mask
        else:
            out_tensor = sum_embeddings / torch.sqrt(sum_mask)
    else:
        raise NotImplementedError
    return out_tensor

