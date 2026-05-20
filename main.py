from core.general import *
from core.log import log as _log

from lib.type import ds
from lib.env import *
from lib.model import get_model, train
from lib.test import run_tests
from lib.encode import encode

from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

log = _log('main')

_f = ds('data/train')
f = DataLoader(_f, batch_size = BATCH_SIZE, shuffle = True)

model = get_model(_f.mark.__len__())

locc = nn.BCEWithLogitsLoss()
optr = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
schr = torch.optim.lr_scheduler.ReduceLROnPlateau(optr, 'min', 0.1, 3, min_lr = 1e-5)

inited = False

for ep in range(EPOCHS):
    if exist(f'data/model.{ep}.mdl'):
        continue
    if not inited and not ep == 0:
        model.load_state_dict(torch.load(f'data/model.{ep - 1}.mdl', weights_only = True))
        log.info('Train', rep = True)(f'Weights loaded from EPOCH {ep - 1}')
        inited = True
    log.info('Train', rep = True)(f'Training EPOCH {ep}')
    
    losses = train(model, f, locc, optr, ep)

    info(f'Finished training EPOCH {ep} with loss {losses / len(f)}, saving')
    log.info('Train')(f'Finished training EPOCH {ep} with loss {losses / len(f)}, saving')
    torch.save(model.state_dict(), f'data/model.{ep}.mdl')

    # schr.step(val_loss)
    # info('Running Tests')
    # encode(run_tests(model), _f.mark)
