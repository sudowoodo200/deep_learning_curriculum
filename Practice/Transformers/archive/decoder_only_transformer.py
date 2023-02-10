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

# import transformer modules
from transformer_sublayers import *

## decoder layer, without encoder source
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        return self.sublayer[1](x, self.feed_forward)

## Decoder
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, tgt_mask):
        for layer in self.layers:
            x = layer(x, tgt_mask)
        return self.norm(x)

## Transformer
class Transformer(nn.Module):

    def __init__(self, decoder, tgt_embed, generator):
        super().__init__()
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def forward(self, x, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(x), tgt_mask))

## Building the model
def make_model(tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    decoder_layer = DecoderLayer(d_model, c(attn), c(ff), dropout)
    decoder = Decoder(decoder_layer, N)
    model = Transformer(decoder, nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)), Generator(d_model, tgt_vocab))

    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model