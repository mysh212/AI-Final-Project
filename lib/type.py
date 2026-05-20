from core.general import *
from core.log import log as _log

from lib.env import device

from dotenv import load_dotenv

import cv2
import albumentations as A

import torch
from torch.utils.data import Dataset

log = _log('type')
load_dotenv()

transformation = A.Compose([
    A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit = .05, scale_limit = .1, rotate_limit = 10, p = 0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.1),
    A.Normalize(),
    # A.pytorch.ToTensorV2(),
])

static_transformation = A.Compose([
    # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    # A.ShiftScaleRotate(shift_limit = .05, scale_limit = .1, rotate_limit = 20, p = 0.5),
    # A.RandomBrightnessContrast(p=0.3),
    # A.GaussianBlur(p=0.2),
    A.Normalize(),
    # A.pytorch.ToTensorV2(),
])

class ds(Dataset):
    def __init__(self, path: str, transformation = transformation):
        log.debug('Dataset Load')(f'Loading datasets from {path}')
        self.path = path
        f = ls(path)
        self.f = []
        self.mark = f.copy()
        self.to = {i: j for j, i in enumerate(f)}
        self.transformation = transformation
        for k, i in enumerate(f):
            log.debug('Dataset Load')(f'Loading datasets from class {i}')
            self.f.extend([[f'{path}/{i}/{j}', k] for j in ls(f'{path}/{i}') if j.endswith('.png')])
        log.debug('Dataset Load')(f'Dynamically loaded {len(self.f)} datas')
        return

    def __getitem__(self, index):
        return torch.tensor(self.transformation(image = cv2.imread(self.f[index][0], cv2.IMREAD_GRAYSCALE))['image']) / 255.0 * 2048 - 1024, self.onehot(self.f[index][1])

    def __len__(self):
        return len(self.f)
    
    def rev(self, x: int):
        return self.mark[x]
    
    @staticmethod
    def one_hot(x: int, len: int, dtype = torch.float32):
        pre = torch.zeros(len, device = device, dtype = dtype)
        pre[x] = 1
        return pre

    def onehot(self, x: int):
        return self.one_hot(x, len(self.mark))
    
class ts(Dataset):
    def __init__(self, path: str, transformation = static_transformation):
        log.debug('Test Dataset Load')(f'Loading datasets from {path}')
        self.path = path
        self.f = [f'{path}/{i}' for i in ls(path)]
        self.transformation = transformation
        log.debug('Test Dataset Load')(f'Dynamically loaded {len(self.f)} datas')
        return

    def __getitem__(self, index):
        return torch.tensor(self.transformation(image = cv2.imread(self.f[index], cv2.IMREAD_GRAYSCALE))['image']) / 255.0 * 2048 - 1024

    def __len__(self):
        return len(self.f)