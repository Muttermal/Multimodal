# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/21 14:07
# @Author  : zhangguangyi
# @File    : utils.py

import os
import time
import torch
import torch.nn as nn


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        """Timer constructor."""
        self.reset()

    def reset(self):
        """Reset timer."""
        self.running = True
        self.total = 0
        self.start = time.time()

    def resume(self):
        """Resume."""
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        """Stop."""
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    @property
    def time(self):
        """Return time."""
        if self.running:
            return self.total + time.time() - self.start
        return self.total


def get_lr(optimizer: torch.optim.Optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_data(path, all_files: list):
    if not os.path.isdir(path):
        raise TypeError(f"The path of {path} is not a directory")
    file_list = os.listdir(path)
    for fn in file_list:
        if not fn.startswith("."):
            cur_path = os.path.join(path, fn)
            if os.path.isdir(cur_path):
                get_data(cur_path, all_files)
            else:
                all_files.append({"root": path, "file_name": fn})
    return all_files


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
