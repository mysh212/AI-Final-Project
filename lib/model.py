from core.general import *
from core.log import log as _log

from lib.env import device, EPOCHS, BATCH_SIZE, RELU

import torchxrayvision as xray

import torch
from torch import nn
from tqdm import tqdm

from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("shreydan/CheXpert-5-convnextv2-tiny-384")
log = _log('model')

def get_model(feature_count: int):
    model = AutoModelForImageClassification.from_pretrained("shreydan/CheXpert-5-convnextv2-tiny-384")
    name, layer = next((n, m) for n, m in list(model.named_modules())[::-1] if isinstance(m, nn.Linear))
    *path, attr = name.split('.'); p = model
    for i in path: p = getattr(p, i)
    setattr(p, attr, nn.Sequential(nn.Linear(layer.in_features, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, feature_count)) if RELU else nn.Sequential(nn.Linear(layer.in_features, feature_count)))
    model.op_threshs = None
    model = model.to(device)

    for ep in range(EPOCHS + 1):
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
        img = img.permute(0, 3, 1, 2).contiguous()
        img = img.to(device)
        ans = ans.to(device)

        # img = processor(images = img)

        optr.zero_grad()
        pred = model(img).logits

        # error(pred, ans)
        loss = locc(pred, ans)
        loss.backward()

        optr.step()
        losses += loss.item()

        bar.set_postfix(EPOCH = ep, loss = losses / (j + 1))

    return losses

def validate(model, g, locc, ep):
    bar = tqdm(g)
    losses = 0.0
    yes = 0

    model.eval()
    with torch.no_grad():
        for j, (img, ans) in enumerate(bar):
            img = img.permute(0, 3, 1, 2).contiguous()
            img = img.to(device)
            ans = ans.to(device)

            pred = model(img).logits

            loss = locc(pred, ans)
            losses += loss.item()

            yes += ((torch.argmax(pred, dim = 1) == (torch.argmax(ans, dim = 1))).sum()).item()

            bar.set_postfix(EPOCH = ep, loss = losses / (j + 1), accuracy = f'{yes} / {(j + 1) * BATCH_SIZE} = {yes / ((j + 1) * BATCH_SIZE):.2f}')
        return losses / len(g), yes
