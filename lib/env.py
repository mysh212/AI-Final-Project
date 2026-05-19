from core.general import *

import os
from dotenv import load_dotenv
import torch

load_dotenv()

BATCH_SIZE = 3
LEARNING_RATE = 1e-3
EPOCHS = 10

_device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device(_device)

info(f'Using {_device} to run')

def env(x: str, default = None):
    return os.getenv(x) or default