from core.general import *
# from core.log import log as _log

from lib.env import device

import torchxrayvision as xray

from torch import nn
from tqdm import tqdm
# log = _log('model')

def get_model(feature_count: int):
    model = xray.models.get_model("densenet121-res224-chex")
    debug(model.classifier.in_features)
    model.classifier = nn.Linear(model.classifier.in_features, feature_count)
    model.op_threshs = None
    model = model.to(device)

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