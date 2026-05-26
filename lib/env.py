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
REVERSE_PENALTY = True

_device = 'cpu'
if SPECIAL_DEVICES:
    _device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(_device)

info(f'Using {_device} to run')

def env(x: str, default = None):
    return os.getenv(x) or default
