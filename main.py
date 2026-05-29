from core.general import *

import re
import numpy as np
import pandas as pd

import torchxrayvision as xrv
import torch, torchvision
from tqdm import tqdm

import cv2

GATE = .5213

# from lib.env import device
device = torch.device('cpu')

def predict(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
    # img = img.mean(2)[None, ...] # Make single color channel
    img = np.array([img])

    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

    img = transform(img)
    img = torch.from_numpy(img).to(device)

    # Load model and process image
    model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
    outputs = model(img[None,...]) # or model.features(img[None,...]) 

    # Print results
    return (dict(zip(model.pathologies,outputs[0].detach().cpu().numpy())))

mark = ls('data/train')

def shape(p):
    name, prob = max([i for i in zip(p.keys(), p.values()) if i[0] in mark], key = lambda x: x[1])
    if prob < GATE:
        name = 'No Finding'
    return name

aans = []
def write(filename: str, ans: str):
    aans.append(dict(filename = filename, label = ans))
    return

filter = re.compile(r'^(?:.*/)?((.+)\.(?:png|jpg))$')

def export():
    df = pd.DataFrame(aans)
    df['id'] = pd.to_numeric(df.filename.map(lambda x: filter.findall(x)[0][1]))
    df['filename'] = df.filename.map(lambda x: filter.findall(x)[0][0])
    df = df.sort_values(['id'])
    print(df)
    return df

for i in tqdm(ls('data/test')):
    write(i, shape(predict(f'data/test/{i}')))

export().to_csv('data/submit.csv', index = False)