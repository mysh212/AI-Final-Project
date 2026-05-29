from core.general import *
from core.log import log as _log

from lib.type import ds
from lib.env import *
from lib.model import get_model, train, validate
from lib.test import run_tests
from lib.encode import encode
from submit import kaggle_submit as auto_submit

from torch import nn
from torch.utils.data import DataLoader, random_split

from collections import defaultdict
import random
import re

log = _log('main')

_ds = ds('data/train')
# _f, _g = random_split(_ds, [.8, .2], torch.Generator().manual_seed(212))

random.seed(212)

filter = re.compile(r'^.*/(\d+)_\d+\.png$')

patient = defaultdict(list)

for i, j in _ds.f:
    label = filter.findall(i)[0]
    patient[label].append([i, j])

patient = list(zip(patient.keys(), patient.values()))

random.shuffle(patient)

cases = []
for i in patient:
    cases.extend(i[1])

cut = int(len(cases) * VALIDATE_RATIO)

_f, _g = ds('data/train', files = cases[:cut], transform = True), ds('data/train', files = cases[cut:], transform = False)

f = DataLoader(_f, batch_size = BATCH_SIZE, shuffle = True)
g = DataLoader(_g, batch_size = BATCH_SIZE, shuffle = False)

model = get_model(_ds.mark.__len__())

locc = nn.CrossEntropyLoss(weight = _f.get_weight(list(range(len(_f.mark)))[::-1] if REVERSE_PENALTY else None).to(device) if WEIGHT_BALANCE else None)
optr = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
schr = torch.optim.lr_scheduler.ReduceLROnPlateau(optr, 'min', factor = 0.1, patience = 3, min_lr = 1e-5)
for ep in range(STATIC_ROUND):
    name, _ = next((n, m) for n, m in list(model.named_modules())[::-1] if isinstance(m, nn.Sequential))
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(name)

    log.info('Train', rep = True)(f'Staticly training EPOCH {ep}')
    losses = train(model, f, locc, optr, ep)

    info(f'Finished training EPOCH {ep} with loss {losses / len(f)}, saving')
    log.info('Train')(f'Finished training EPOCH {ep} with loss {losses / len(f)}, saving')
    torch.save(model.state_dict(), f'data/model.{ep}.mdl')

for p in model.parameters():
    p.requires_grad = True

for ep in range(EPOCHS):
    if exist(f'data/model.{ep}.mdl'):
        continue

    log.info('Train', rep = True)(f'Training EPOCH {ep}')

    losses = train(model, f, locc, optr, ep)

    info(f'Finished training EPOCH {ep} with loss {losses / len(f)}, saving')
    log.info('Train')(f'Finished training EPOCH {ep} with loss {losses / len(f)}, saving')
    torch.save(model.state_dict(), f'data/model.{ep}.mdl')

    log.info('Validate', rep = True)(f'Validating for EPOCH {ep}')
    val, yes = validate(model, g, locc, ep)
    log.info('Validate')(val = val, yes = yes)

    # info('Running Tests')
    # encode(run_tests(model), _ds.mark)

    log.info('Submit', rep = True)(f'Submitting EPOCH {ep}')
    auto_submit(describe(EPOCH = ep, LEARNING_RATE = optr.param_groups[0]["lr"]))

    schr.step(val) # (1 - (yes / len(_g)))
    log.info('scheduler', rep = True)(f'New learning rate = {optr.param_groups[0]["lr"]}')