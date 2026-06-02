from core.general import *
from core.log import log as _log

from lib.env import BATCH_SIZE, TTA_TIMES, device
from lib.type import ts

import torch
from torch.utils.data import DataLoader

import pandas as pd
from tqdm import tqdm

_f = ts('data/test')
f = DataLoader(_f, batch_size = BATCH_SIZE)

log = _log('test')

def run_tests(model) -> dict:
    aans = {}
    log.info('Run Tests')(f'Start running tests')
    for img, path in tqdm(f):
        img = img.permute(0, 3, 1, 2).contiguous()
        img = img.to(device)

        ans = model(img)
        for _ in range(TTA_TIMES):
            img = _f.batch_varint(img)
            ans += model(img)

        # pre = pre / (TTA_TIMES + 1)
        for i, j in zip(path, torch.argmax(ans, dim = 1)):
            aans[i] = j.item()

    log.info('Run Tests')(f'Test Finished.')
    return aans
