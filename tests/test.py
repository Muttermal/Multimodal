# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/28 17:43
# @Author  : zhangguangyi
# @File    : test.py
import argparse
import logging
from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
import transformers

from Multimodal.Model import ViTAEWindowNoShiftBasic, SwinTransformer
from Multimodal.DataLoader import make_dataloader

logging.basicConfig(level=logging.INFO)
model_type = "swin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_type.lower() == "swin":
    model = SwinTransformer(
        img_size=224, patch_size=4, in_chans=3, num_classes=10, embed_dim=96,
        depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.,
        qkv_bias=True, qk_scale=0.5, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
        norm_layer=torch.nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False
    )
    pretrained_dict = torch.load("../../../pre_trained_model/CV/rsp-swin-t-ckpt.pth", map_location="cpu")["model"]
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("head")}  # 去掉分类头
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)    # model.to()为in-place方法

elif model_type.lower() == "vitae":
    """
    model structure: Reduction cell + downsample + Normal cell + linear
    params:
        img_size: int 图像尺寸，默认为224
        in_channels:int 卷积的输入维度，rgb3维
        stages: int base layer的层数，默认为4层(一个RC + 一个NC为一层，其中RC里面包含卷积操作，NC里面包含attention操作)
        embed_dims: int or list 每一层中RC卷积出来的维度，默认分别为[64, 64, 64, 64]
        RC_tokens_type: list
        NC_tokens_type
        token_dims
        downsample_ratios:采样比率, list
        NC_depth
        NC_heads
        RC_heads
        mlp_ratio
        NC_group
        RC_group
        img_size
        window_size
        drop_path_rate
        frozen_stages
        norm_eval
        pretrained

    """
    model = ViTAEWindowNoShiftBasic(img_size=1024, stages=4, embed_dims=[64, 64, 128, 256],
                                    token_dims=[64, 128, 256, 512], downsample_ratios=[4, 2, 2, 2],
                                    RC_heads=[1, 1, 2, 4], NC_heads=[1, 2, 4, 8],
                                    RC_tokens_type=['swin', 'swin', 'transformer', 'transformer'],
                                    NC_tokens_type=['swin', 'swin', 'transformer', 'transformer'],
                                    RC_group=[1, 16, 32, 64], NC_group=[1, 32, 64, 128], NC_depth=[2, 2, 8, 2],
                                    mlp_ratio=4., drop_path_rate=0.3, window_size=7,
                                    pretrained="/home/nlp/zhangguangyi/pre_trained_model/CV/rsp-vitaev2-s-ckpt.pth",
                                    frozen_stages=-1, norm_eval=False)
else:
    raise KeyError("model not supported")
logging.info("load model successful")

parser = argparse.ArgumentParser(description="test")
parser.add_argument("--data_root", type=str, help="data root", default="../data/2750")
parser.add_argument("--size", type=int, default=224, help="input dimension of image")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--val_batch_size", type=int, default=64)
parser.add_argument("--val_ratio", type=float, default=0.2)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--weight_decay", type=float, default=5e-5)
parser.add_argument("--label_dict", type=str, default="../data/2750/label_dict.json")

args = parser.parse_args()

train_dataloader,  val_dataloader = make_dataloader(args, data_type="image")
total_steps = len(train_dataloader) * args.epochs
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': args.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200,
                                                         num_training_steps=total_steps)

criterion = torch.nn.CrossEntropyLoss()
for batch in train_dataloader:
    inputs, target = batch["image"].to(device), batch["label"].to(device)  # tensor.to()不是in-place方法
    model.train()
    logit = model(inputs)   # input=[batch_size, channel, H, W]
    loss = criterion(logit, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()


input_data = Variable(torch.rand(16, 3, 224, 224))
writer = SummaryWriter(log_dir='./log', comment='vitae')
with writer:
    writer.add_graph(model, (input_data,))
