from core.general import *
from core.log import log as _log

from lib.type import ds
from lib.env import *
from lib.model import get_model, train, validate
from lib.test import run_tests
from lib.encode import encode

from torch import nn
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

log = _log('main')

_ds = ds('data/train')
_f, _g = random_split(_ds, [.8, .2], torch.Generator().manual_seed(212))
f = DataLoader(_f, batch_size = BATCH_SIZE, shuffle = True)
g = DataLoader(_g, batch_size = BATCH_SIZE, shuffle = False)

model = get_model(_ds.mark.__len__())

locc = nn.BCEWithLogitsLoss()
optr = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
schr = torch.optim.lr_scheduler.ReduceLROnPlateau(optr, 'min', 0.1, 3, min_lr = 1e-5)

for ep in range(EPOCHS):
    if exist(f'data/model.{ep}.mdl'):
        continue
    log.info('Train', rep = True)(f'Training EPOCH {ep}')
    
    losses = train(model, f, locc, optr, ep)

    info(f'Finished training EPOCH {ep} with loss {losses / len(f)}, saving')
    log.info('Train')(f'Finished training EPOCH {ep} with loss {losses / len(f)}, saving')
    torch.save(model.state_dict(), f'data/model.{ep}.mdl')
    
    log.info('Validate', rep = True)(f'Validating for EPOCH {ep}')
    val, yes = validate(model, g, ep)
    log.info('Validate')(val = val, yes = yes)

    schr.step(1 - (yes / len(_g)))
    log.info('scheduler', rep = True)(f'New learning rate = {optr.param_groups[0]["lr"]}')
    # info('Running Tests')
    # encode(run_tests(model), _ds.mark)
