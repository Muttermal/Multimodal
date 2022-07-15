# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/20 15:02
# @Author  : zhangguangyi
# @File    : dataloader.py

from functools import partial
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


from .dataset import TextDataSet, ImageDataSet, VideoDataSet, AudioDataSet


TYPE2DATA = {
    "text": TextDataSet,
    "image": ImageDataSet,
    "video": VideoDataSet,
    "audio": AudioDataSet,
}


def make_dataloader(args, data_type: str = "text"):
    data_type = data_type.lower()
    if data_type not in TYPE2DATA:
        raise TypeError("only data type in the following formats is supported:{text, image, video, audio}")

    current_dataset = TYPE2DATA[data_type](args)
    size = len(current_dataset)
    val_size = int(size * args.val_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(current_dataset, [size - val_size, val_size],
                                                               generator=torch.Generator().manual_seed(2022))

    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    # train_sampler = BalanceSampling(data_source=train_dataset, batch_size=args.batch_size,
    #                                 max_steps=args.max_steps, drop_last=True)
    # train_dataloader = dataloader_class(train_dataset, batch_sampler=train_sampler)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    return train_dataloader, val_dataloader
