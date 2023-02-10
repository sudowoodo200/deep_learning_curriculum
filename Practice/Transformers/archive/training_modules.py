# import pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# import other modules
import numpy as np
import matplotlib.pyplot as plt
import time, math, os, sys, argparse, random, copy

# import transformer
from decoder_only_transformer import *

## Simple loss computation
class SimpleLossCompute:

    "A simple loss compute and train function."
    def __init__(self, generator, criterion, optimizer=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = optimizer

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss.data.item() * norm

## Data Batching and Masking
class Batch:
    
        "Object for holding a batch of data with mask during training."
        def __init__(self, x, pad=0):
            self.data = x
            self.mask = self.make_mask(x, pad)
    
        @staticmethod
        def make_mask(x, pad):
            "Create a mask to hide padding and future words."
            def subsequent_mask(size):
                "Mask out subsequent positions."
                attn_shape = (1, size, size)
                subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
                return torch.from_numpy(subsequent_mask) == 0
            mask = (x != pad).unsqueeze(-2)
            mask = mask & Variable(subsequent_mask(x.size(-1)).type_as(mask.data))
            return mask

## Training
def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, (x, y) in enumerate(data_iter):
        out = model.forward(x, y)
        loss = loss_compute(out, y, 1)
        total_loss += loss
        total_tokens += y.size(0) * y.size(1)
        tokens += y.size(1)
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / y.size(1), tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

## Wrap the Adam Optimizer
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


## Label Smoothing
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))