# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/13 15:03
# @File    : NSE_loss.py

import torch
from torch import nn
import torch.nn.functional as F


class NSELoss(nn.Module):
    """
    """

    __constants__ = ['alpha']

    def __init__(self, is_supervised: bool = True, alpha: float = 0.05):
        super().__init__()
        self.is_supervised = is_supervised
        self.alpha = alpha

    def supervised(self, logit: torch.Tensor, target: torch.Tensor):
        """
        Args:
            logit:batch内样本的embedding表示，batch内正负样本分布为[x, x+, x-, ...]
            target:
        Returns:
        """
        target = torch.arange(logit.shape[0], device=logit.device)
        use_row = torch.where((target + 1) % 3 != 0)[0]
        target = (use_row - use_row % 3 * 2) + 1
        # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
        sim = F.cosine_similarity(logit.unsqueeze(1), logit.unsqueeze(0), dim=-1)
        # 将相似度矩阵对角线置为很小的值, 消除自身的影响
        sim = sim - torch.eye(logit.shape[0], device=logit.device) * 1e12
        # 选取有效的行
        sim = torch.index_select(sim, 0, use_row)
        # 相似度矩阵除以温度系数
        sim = sim / self.alpha
        # 计算相似度矩阵与y_true的交叉熵损失
        loss = F.cross_entropy(sim, target)
        return loss

    def unsupervised(self, logit: torch.Tensor, target: torch.Tensor):
        """
        Args:
            logit: batch内样本的embedding表示，batch内样本分布为[a,a,b,b,c,c,...]
            target:
        Returns:
        """
        target = torch.arange(logit.shape[0], device=logit.device)
        target = (target - target % 2 * 2) + 1
        # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
        sim = F.cosine_similarity(logit.unsqueeze(1), logit.unsqueeze(0), dim=-1)
        # 将相似度矩阵对角线置为很小的值, 消除自身的影响
        sim = sim - torch.eye(logit.shape[0], device=logit.device) * 1e12
        # 相似度矩阵除以温度系数
        sim = sim / self.alpha
        # 计算相似度矩阵与y_true的交叉熵损失
        loss = F.cross_entropy(sim, target)
        return loss
        pass

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if self.is_supervised:
            loss = self.supervised(y_pred, y_true)
        else:
            loss = self.unsupervised(y_pred, y_true)
        return loss
