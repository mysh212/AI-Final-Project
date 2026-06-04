from core.general import *
from core.log import log as _log

from lib.env import device, TRANSFORM, N

from dotenv import load_dotenv

import re
from re import Match

from typing import cast, Sequence

import cv2
import albumentations as A
from albumentations import Compose

import torch
from torch.utils.data import Dataset
import numpy as np

log = _log('type')
load_dotenv()

transformation = A.Compose([
    A.Resize(N, N),
    A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, scale=(0.9, 1.1), rotate=(-10, 10), p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.1),
    # A.Normalize(),
    # A.pytorch.ToTensorV2(),
])

static_transformation = A.Compose([
    A.Resize(N, N),
    # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    # A.ShiftScaleRotate(shift_limit = .05, scale_limit = .1, rotate_limit = 20, p = 0.5),
    # A.RandomBrightnessContrast(p=0.3),
    # A.GaussianBlur(p=0.2),
    # A.Normalize(),
    # A.pytorch.ToTensorV2(),
])

class ds(Dataset):
    def __init__(self, path: str, files: list | None = None, transform: Compose | bool = transformation if TRANSFORM else static_transformation):
        log.debug('Dataset Load')(f'Loading datasets from {path}')
        if isinstance(transform, bool):
            transform = transformation if transform else static_transformation
        self.path = path
        f = sorted(ls(path))
        self.f = []
        self.mark = f
        self.to = {i: j for j, i in enumerate(f)}
        self.transformation = transform
        
        if files is not None:
            self.f = files
        else:
            self.f = []
            for k, i in enumerate(f):
                log.debug('Dataset Load')(f'Loading datasets from class {i}')
                class_path = f'{path}/{i}'
                self.f.extend([[f'{path}/{i}/{j}', k] for j in ls(class_path) if j.endswith('.png')])
        
        log.debug('Dataset Load')(f'Dynamically loaded {len(self.f)} datas')
        return
    
    def get_weight(self, op: Sequence[float | int] | torch.Tensor | None = None) -> torch.Tensor:
        mark = [0 for _ in range(len(self.mark))]
        for _, j in self.f:
            mark[j] += 1
        return torch.tensor([1 / (i + 1e-6) for i in mark]) * (torch.tensor(op) if op is not None else torch.ones(len(self.mark)))

    def __getitem__(self, index):
        image = cv2.imread(self.f[index][0])
        image = self.transformation(image=image)['image']
        image = (image.astype(np.float32) / 255.0 * 2048) - 1024
        return torch.tensor(image), self.onehot(self.f[index][1])

    def __len__(self):
        return len(self.f)
    
    def rev(self, x: int):
        return self.mark[x]
    
    @staticmethod
    def hot_one(x: torch.Tensor) -> int:
        for j, i in enumerate(x):
            if i != 0:
                return j
        return len(x)
    
    @staticmethod
    def one_hot(x: int, len: int, dtype = torch.float32):
        pre = torch.zeros(len, dtype = dtype)
        pre[x] = 1
        return pre

    def onehot(self, x: int):
        return self.one_hot(x, len(self.mark))
    
class ts(Dataset):
    def __init__(self, path: str, transform: bool = False):
        log.debug('Test Dataset Load')(f'Loading datasets from {path}')
        self.path = path
        self.f = [f'{path}/{i}' for i in ls(path)]
        self.transformation = transformation if transform else static_transformation
        log.debug('Test Dataset Load')(f'Dynamically loaded {len(self.f)} datas')
        return

    def _read_img(self, path: str) -> np.ndarray:
        image = cv2.imread(path)
        image = self.transformation(image=image)['image']
        image = (image.astype(np.float32) / 255.0 * 2048) - 1024
        return image

    def __getitem__(self, index):
        return torch.tensor(self._read_img(self.f[index])), cast(Match[str], re.match(r'^.*/(\d+)\..+$', self.f[index])).groups()[0]

    @staticmethod
    def get_variant(img):
        return transformation(image = img)['image']
    
    @staticmethod
    def batch_varint(imgs):
        imgs = (imgs + 1024) / 2048 * 255
        imgs = imgs.clone().permute(0, 2, 3, 1)
        tmp = imgs.cpu().numpy()
        for i in range(tmp.shape[0]):
            tmp[i] = transformation(image = tmp[i])['image']
        tmp = (tmp.astype(np.float32) / 255.0 * 2048) - 1024
        return torch.tensor(tmp, device = device).permute(0, 3, 1, 2)

    def variant(self, index):
        return self.get_variant(self._read_img(self.f[index]))

    def __len__(self):
        return len(self.f)
