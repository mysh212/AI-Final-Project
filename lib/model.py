from core.general import *
from core.log import log as _log

from lib.env import device, EPOCHS, BATCH_SIZE, RELU

# import torchxrayvision as xray
from source.model import DenseNet121 as mds

import torch
from torch import nn
from tqdm import tqdm
import torchvision

log = _log('model')

def get_model(feature_count: int):
#    if not exist('data/model.0.mdl'):
#        error('Please run `init.sh` first.')
#        quit(1)

    model = mds(feature_count).to(device)
    
    checkpoint = torch.load('source/model.pth.tar', map_location = device)
    state_dict = checkpoint['state_dict']
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        
        name = name.replace('.norm.1.', '.norm1.')
        name = name.replace('.norm.2.', '.norm2.')
        name = name.replace('.conv.1.', '.conv1.')
        name = name.replace('.conv.2.', '.conv2.')
        
        if name in model.state_dict() and model.state_dict()[name].shape != v.shape:
            continue

        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict = False)

    for ep in range(EPOCHS + 1):
        if not exist(f'data/model.{ep}.mdl'):
            if ep == 0:
                break
            model.load_state_dict(torch.load(f'data/model.{ep - 1}.mdl', weights_only = True, map_location = device))
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
        pred = model(img)

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

            pred = model(img)

            loss = locc(pred, ans)
            losses += loss.item()

            yes += ((torch.argmax(pred, dim = 1) == (torch.argmax(ans, dim = 1))).sum()).item()

            bar.set_postfix(EPOCH = ep, loss = losses / (j + 1), accuracy = f'{yes} / {(j + 1) * BATCH_SIZE} = {yes / ((j + 1) * BATCH_SIZE):.2f}')
        return losses / len(g), yes
