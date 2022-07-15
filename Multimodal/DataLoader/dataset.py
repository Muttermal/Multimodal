# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/20 15:11
# @Author  : zhangguangyi
# @File    : dataset.py

import cv2
import json
import logging
import os
from pathlib import Path
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


from .data_augmentation import get_transforms
from ..utils import get_data

logging.basicConfig(level=logging.INFO)


class BaseDataSet(Dataset):

    def __init__(self, type_):
        self._type = type_

    def __repr__(self):
        return f"The type of current dataset is {self._type}"

    def __getitem__(self, item):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class TextDataSet(BaseDataSet):

    def __init__(self, args):
        super(TextDataSet, self).__init__(type_="Text")
        self.data_path = Path(args.text_data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.pre_train_model_dir)
        self.max_seq_len = args.max_seq_len
        self.text_data = self.get_data()
        if args.label_dict:
            self.label_dict = json.load(open(args.label_dict, "r", encoding="utf-8"))
            self.label_dict_env = {v: k for k, v in self.label_dict.items()}

    def tokenize_text(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=self.max_seq_len, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        token_type_ids = torch.zeros_like(input_ids)
        return input_ids, mask, token_type_ids

    def get_data(self):
        all_data = {"text": [], "label": []}
        with open(self.data_path, "r", encoding="utf-8") as ft:
            for line in ft:
                text, label = line.rstrip("\n").split("\t")
                all_data["text"].append(text)
                all_data["label"].append(label)
        return all_data

    def __iter__(self):
        for i in super().__iter__():
            yield i

    def __getitem__(self, idx: int):
        text = self.text_data["text"][idx]
        label = self.text_data["label"][idx]
        # label_id = np.ones(len(self.label_dict))
        # label_id[self.label_dict_env[label]] = 1  # one-hot 形式
        label_id = self.label_dict_env[label]
        input_ids, attention_mask, token_type_ids = self.tokenizer(text)
        data = dict(
            text=text,
            label_id=label_id,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return data

    def make_prompt(self, prompt_tokens: dict):
        # TODO
        pass

    def __len__(self):
        return len(self.text_data)


class ImageDataSet(BaseDataSet):

    def __init__(self, args):
        super(ImageDataSet, self).__init__(type_="Image")
        self.data_path = args.data_root
        self.transforms = get_transforms(args, mode="train")
        self.file_names = get_data(self.data_path, all_files=[])
        if args.label_dict:
            self.label_dict = json.load(open(args.label_dict, "r", encoding="utf-8"))
            self.label_dict_env = {v: k for k, v in self.label_dict.items()}

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx: int):
        item = {}
        file_path = os.path.join(self.file_names[idx]["root"], self.file_names[idx]["file_name"])
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['path'] = file_path
        label = self.file_names[idx]["label"].split("_")[0]
        item["label"] = self.label_dict[label]
        return item

    def __iter__(self):
        pass


class VideoDataSet(object):

    def __init__(self, args):
        super(VideoDataSet, self).__init__(type_="Video")
        self.path = args.video_path
        self.frame_number = args.frame_number
        self.frame_path = args.frame_path
        self.frame_file_names = self.video2frame()  # 将视频数据根据帧数提取成图像数据
        self.transforms = get_transforms(args, mode="train")

    def video2frame(self):
        all_frame_file_path = []
        has_converted = False
        for video_file in os.listdir(self.path):
            frame_saved_path = os.path.join(self.frame_path, video_file)
            if os.path.exists(frame_saved_path) and (not os.listdir(frame_saved_path)):
                has_converted = True
                continue
            else:
                cap = cv2.VideoCapture(os.path.join(self.path, video_file))  # 获取一个视频并打开
                os.makedirs(frame_saved_path, exist_ok=True)
                if cap.isOpened():  # VideoCapture对象是否成功打开
                    fps = cap.get(cv2.CAP_PROP_FPS)  # 返回视频的fps--帧率
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 返回视频的宽
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 返回视频的高
                    logging.info("file: ", video_file, 'fps: ', fps, 'width: ', width, 'height: ', height)
                    i = 0
                    while i <= self.frame_number:
                        i += 1
                        ret, frame = cap.read()  # 读取一帧视频
                        # ret 读取了数据就返回True,没有读取数据(已到尾部)就返回False
                        # frame 返回读取的视频数据--一帧数据
                        file_name = os.path.join(frame_saved_path, f"{i}_frame.jpg")
                        cv2.imwrite(file_name, frame)
                all_frame_file_path.append(video_file)
        if has_converted:
            logging.info("videos frames files existed")
        else:
            logging.info("all videos have been extracted to frames")
        return all_frame_file_path

    def __len__(self):
        return len(self.frame_file_names)

    def __getitem__(self, idx: int):
        item = {}
        video_name = self.frame_file_names[idx]
        item["label"] = video_name
        file_root = os.path.join(self.frame_path, video_name)
        frame_tensor = []
        for file_name in os.listdir(file_root):
            file_path = os.path.join(file_root, file_name)
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transforms(image=image)['image']
            frame_tensor.append(torch.tensor(image).permute(2, 0, 1).float())   # [channel, H, W]
        item["tensor"] = torch.stack(frame_tensor)  # [frame_nums, channel, H, W]
        return item

    def __iter__(self):
        pass


class AudioDataSet(object):
    def __init__(self, args):
        super(AudioDataSet, self).__init__(type_="Audio")
        self.path = args.audio_path
        self.audio_file = os.listdir(self.path)
        self.frame_number = args.frame_number

    def __len__(self):
        return len(self.audio_file)

    def __getitem__(self, idx: int):
        item = {}
        curr_audio = self.audio_file[idx]
        item["label"] = curr_audio
        data, sample_rate = sf.read(file=os.path.join(self.path, curr_audio),
                                    dtype="float32", frames=self.frame_number, start=0)
        item["audio"] = data    # [frames x channels]
        return item

    def __iter__(self):
        pass
