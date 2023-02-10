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
def get_batch(x, y, batch_size, i = None):

    # random sampling v.s. sequential sampling
    if i is None:
        index = torch.randint(0, len(x), (batch_size, ))
    else:
        index = torch.arange(i, i+batch_size)

    xb = torch.stack([x[i] for i in index])
    yb = torch.stack([y[i] for i in index])
    return xb, yb


## Decoding Logits to vocab embedding
def decode_logits(logits, lookback_block_size = 1):

    batch_size, block_length, vocab_size = logits.shape

    logits_pred = logits[:, -lookback_block_size : , :] # (batch_size, lookback_block_size, vocab_size)
    probs = F.softmax(logits_pred, dim=-1) # (batch_size, lookback_block_size, vocab_size) where sum over vocab_size = 1
    probs = probs.reshape(batch_size * lookback_block_size, -1) # (batch_size * lookback_block_size, vocab_size
    output = torch.multinomial(probs, num_samples = 1) # (batch_size, lookback_block_size, 1)
    output = output.reshape(batch_size, -1) # (batch_size, lookback_block_size)

    return output

## Decoding vocab embeddings to text
def decode_vocab(x, decoder):
    
    # apply decoder to each row
    # breakpoint()
    x = x.tolist()
    x = [decoder(row) for row in x]
    x = pd.DataFrame(x)
    
    return x