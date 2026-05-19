from core.general import *
from core.log import log as _log

from lib.env import device

from dotenv import load_dotenv

import cv2

import torch
from torch.utils.data import Dataset

log = _log('type')
load_dotenv()

class ds(Dataset):
    def __init__(self, path: str):
        log.debug('Dataset Load')(f'Loading datasets from {path}')
        self.path = path
        f = ls(path)
        self.f = []
        self.mark = f.copy()
        self.to = {i: j for j, i in enumerate(f)}
        for k, i in enumerate(f):
            log.debug('Dataset Load')(f'Loading datasets from class {i}')
            self.f.extend([[f'{path}/{i}/{j}', k] for j in ls(f'{path}/{i}') if j.endswith('.png')])
        log.debug('Dataset Load')(f'Dynamically loaded {len(self.f)} datas')
        return

    def __getitem__(self, index):
        return torch.tensor(cv2.imread(self.f[index][0], cv2.IMREAD_GRAYSCALE)) / 255.0, self.onehot(self.f[index][1])

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
        return self.one_hot(x, len(self.f))