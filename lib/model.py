from core.general import *
from core.log import log as _log

from lib.env import device, EPOCHS, BATCH_SIZE

import torchxrayvision as xray

import torch
from torch import nn
from tqdm import tqdm
log = _log('model')

def get_model(feature_count: int):
    model = xray.models.get_model("densenet121-res224-chex")
    debug(model.classifier.in_features)
    model.classifier = nn.Linear(model.classifier.in_features, feature_count)
    model.op_threshs = None
    model = model.to(device)

    for ep in range(EPOCHS):
        if not exist(f'data/model.{ep}.mdl'):
            if ep == 0:
                break
            model.load_state_dict(torch.load(f'data/model.{ep - 1}.mdl', weights_only = True))
            log.info('Model Load', rep = True)(f'Weights loaded from EPOCH {ep - 1}')
            break

    return model

def train(model, f, locc, optr, ep):
    bar = (tqdm(f))
    losses = 0.0

    model.train()
    for j, (img, ans) in enumerate(bar):
        img = img.unsqueeze(1)
        img = img.to(device)
        
        optr.zero_grad()
        pred = model(img)

        loss = locc(pred, ans)
        loss.backward()

        optr.step()
        losses += loss.item()

        bar.set_postfix(EPOCH = ep, loss = losses / (j + 1))

    return losses

def validate(model, g, ep):
    bar = tqdm(g)
    losses = 0.0
    yes = 0

    locc = nn.BCEWithLogitsLoss()

    model.eval()
    with torch.no_grad():
        for j, (img, ans) in enumerate(bar):
            img = img.unsqueeze(1)
            img = img.to(device)

            pred = model(img)

            loss = locc(pred, ans)
            losses += loss.item()

            yes += ((torch.argmax(pred, dim = 1) == (torch.argmax(ans, dim = 1))).sum())

            bar.set_postfix(EPOCH = ep, loss = losses / (j + 1), accuracy = f'{yes} / {(j + 1) * BATCH_SIZE} = {yes / ((j + 1) * BATCH_SIZE):.2f}')
        return losses / len(g), yes