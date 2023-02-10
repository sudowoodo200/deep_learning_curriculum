import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd, os, json, requests as req, string, numpy as np, argparse


class SelfAttentionHead(nn.Module):
    
    # Note about head_size:
    ##  recall that the input to this module is a tensor of shape (batch_size, block_length, channel_size)
    ##  and the output is a tensor of shape (batch_size, block_length, head_size)
    ##  the head_size is a hyperparameter that you can tune, but is usually set to channel_size // num_heads to keep dimensionality constant
    
    def __init__(self, block_length, channel_size, head_size, dropout=0.1):

        super().__init__()
        self.head_size = head_size
        self.dropout = nn.Dropout(dropout)

        self.q = nn.Linear(channel_size, head_size, bias=False)
        self.k = nn.Linear(channel_size, head_size, bias=False)
        self.v = nn.Linear(channel_size, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_length, block_length)))

    def forward(self, x):

        # breakpoint()
        query = self.q(x) # (batch_size, block_length, head_size)
        key = self.k(x) # (batch_size, block_length, head_size)
        value = self.v(x) # (batch_size, block_length, head_size)

        z = torch.matmul(query, key.transpose(-2,-1)) * ( key.shape[-1] ** -0.5 ) # (batch_size, block_length, block_length)
        z = z.masked_fill(self.tril == 0, float('-inf')) # (batch_size, block_length, block_length)
        z = F.softmax(z, dim=-1) # (batch_size, block_length, block_length)
        z = self.dropout(z) # (batch_size, block_length, block_length)
    
        out = z @ value # (batch_size, block_length, head_size)

        return out
    

class MultiHeadAttention(nn.Module):
        
    def __init__(self, block_length, channel_size, num_heads, dropout=0.1):
        
        super().__init__()
        self.num_heads = num_heads
        self.head_size = channel_size // num_heads
        self.heads = nn.ModuleList([SelfAttentionHead(block_length, channel_size, self.head_size) for _ in range(self.num_heads)])
        self.proj = nn.Linear(self.num_heads * self.head_size, channel_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        # breakpoint()
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (batch_size, block_length, num_heads * head_size)
        out = self.proj(out) # (batch_size, block_length, channel_size)
        out = self.dropout(out) # (batch_size, block_length, channel_size)
        
        return out
    

class FeedForward(nn.Module):
    
    def __init__(self, channel_size, dropout=0.1):
        
        super().__init__()
        self.net = nn.Sequential( nn.Linear(channel_size, channel_size * 4), nn.ReLU(), 
                                 nn.Linear(channel_size * 4, channel_size), nn.Dropout(dropout) )
        
    def forward(self, x):
        
        # breakpoint()
        return self.net(x)


class AttentionBlock(nn.Module):
        
    def __init__(self, block_length, channel_size, num_heads, dropout=0.1):
        
        super().__init__()
        self.multi_attn = MultiHeadAttention(block_length, channel_size, num_heads, dropout)
        self.ff = FeedForward(channel_size, dropout)
        self.layer_norm_1 = nn.LayerNorm(channel_size)
        self.layer_norm_2 = nn.LayerNorm(channel_size)
        
    def forward(self, x):
        
        # breakpoint()
        out = self.layer_norm_1(x + self.multi_attn(x)) # (batch_size, block_length, channel_size)
        out = self.layer_norm_2(out + self.ff(out)) # (batch_size, block_length, channel_size)
        
        return out