from core.general import *
from core.log import log as _log

from lib.env import BATCH_SIZE, device
from lib.type import ts

import torch
from torch.utils.data import DataLoader

import pandas as pd
from tqdm import tqdm

_f = ts('data/test')
f = DataLoader(_f, batch_size = BATCH_SIZE)

log = _log('test')

def run_tests(model):
    aans = []
    log.info('Run Tests')(f'Start running tests')
    for img in tqdm(f):
        img = img.unsqueeze(1)
        img = img.to(device)

        ans = model(img)
        aans.extend(torch.argmax(ans, dim = 1))

    log.info('Run Tests')(f'Test Finished.')
    return aans