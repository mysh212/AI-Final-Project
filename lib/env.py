from core.general import *

import os
from dotenv import load_dotenv
import torch

load_dotenv()

BATCH_SIZE = 10
LEARNING_RATE = 1e-4
EPOCHS = 150
SPECIAL_DEVICES = True
TRANSFORM = True
VALIDATE_RATIO = .7
WEIGHT_BALANCE = True
REVERSE_PENALTY = False
STATIC_ROUND = 5
RELU = False
LABEL = 'Ver 3.0 - Branch CheXNet - Linear'
TTA_TIMES = 4

_device = 'cpu'
if SPECIAL_DEVICES:
    _device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(_device)

info(f'Using {_device} to run')

def env(x: str, default = None):
    return os.getenv(x) or default

def describe(**other) -> str:
    other = dict(other)
    ot = '\n'.join([f'{i} -> {j}' for i, j in zip(other.keys(), other.values())])
    return f'''

BATCH_SIZE = {BATCH_SIZE}
INITIAL_LEARNING_RATE = {LEARNING_RATE}
TRANSFORM = {TRANSFORM}
VALIDATE_RATIO = {VALIDATE_RATIO}
WEIGHT_BALANCE = {WEIGHT_BALANCE}
REVERSE_PENALTY = {REVERSE_PENALTY}
STATIC_ROUND = {STATIC_ROUND}
RELU = {RELU}
LABEL = {LABEL}
TTA_TIMES = {TTA_TIMES}
{ot}'''.strip()
