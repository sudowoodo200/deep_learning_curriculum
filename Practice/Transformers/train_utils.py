import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os, json, requests as req, string
import argparse

## Training / Validation split
def split_data(df, test_size = 0.2):
    train, test = train_test_split(df, test_size=test_size)
    return train, test

## Convert to tensor
def to_tensor(df):
    x = torch.tensor(np.array(df.x.tolist()))
    y = torch.tensor(np.array(df.y.tolist()))
    return x, y

## Get batch
def get_batch(x, y, batch_size):
    index = torch.randint(0, len(x), (batch_size, ))
    xb = torch.stack([x[i] for i in index])
    yb = torch.stack([y[i] for i in index])
    return xb, yb