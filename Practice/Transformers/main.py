import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd, os, json, requests as req, string, numpy as np, argparse

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_util import *
from train_utils import *


## Model Definition
class DecoderModel(nn.Module):

    def __init__(self, vocab_size, channel_size, block_length):

        super().__init__()
        self.channel_embedding = nn.Embedding(vocab_size, channel_size)
        self.positional_encoding = nn.Embedding(block_length, channel_size)
        self.main_block = nn.Linear( block_length * channel_size, block_length * channel_size) # placeholder
        self.ln_final = nn.LayerNorm(channel_size)
        self.out = nn.Linear(channel_size, vocab_size)

    def forward(self, x, y = None):

        batch_size, block_length = x.shape
        token_embedding = self.channel_embedding(x)  # (batch_size, block_length, channel_size)
        positional_embedding = self.positional_encoding(torch.arange(block_length).repeat(batch_size))  # (batch_size, block_length, channel_size)
        embedding = token_embedding + positional_embedding  # (batch_size, block_length, channel_size)
        
        # placeholder
        z = embedding.view(batch_size, -1) # (batch_size, block_length * channel_size)
        z = self.main_block(z) # (batch_size, block_length * channel_size)
        embedding = z.view(batch_size, block_length, -1) # (batch_size, block_length, channel_size)

        # output
        embedding = self.ln_final(embedding) # (batch_size, block_length, channel_size)
        output = self.out(embedding) # (batch_size,, block_length, vocab_size)

        # compute loss if applicable
        if y is not None:
            loss = self.compute_loss(output, y)
        else:
            loss = None

        return output, loss

    def generate(self):
        raise NotImplementedError

    def compute_loss(self, output, y):

        batch_size, block_length, vocab_size = output.shape
        logits = output.view(-1, vocab_size) # (batch_size * block_length, vocab_size)
        target = y.view(-1) # (batch_size * block_length)
        loss = F.cross_entropy(logits, target, ignore_index=0)

        return loss

## Training
def train(model, optimizer, criterion, data, device):

    raise NotImplementedError

    model.train()
    total_loss = 0
    for i in range(0, len(data), batch_size):
        optimizer.zero_grad()
        batch = data[i:i+batch_size]
        batch = [torch.tensor(x, dtype=torch.long, device=device) for x in batch]
        batch = torch.stack(batch, dim=0)
        input = batch[:, :-1]
        target = batch[:, 1:]
        hidden = model.initHidden().to(device)
        output, hidden = model(input, hidden)
        loss = criterion(output.view(-1, output.shape[-1]), target.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data)

## Evaluation
def evaluate(model, data, device):
    
    raise NotImplementedError

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch = [torch.tensor(x, dtype=torch.long, device=device) for x in batch]
            batch = torch.stack(batch, dim=0)
            input = batch[:, :-1]
            target = batch[:, 1:]
            hidden = model.initHidden().to(device)
            output, hidden = model(input, hidden)
            loss = criterion(output.view(-1, output.shape[-1]), target.view(-1))
            total_loss += loss.item()

    return total_loss / len(data)

## Main
if __name__ == '__main__':

    argparse = argparse.ArgumentParser()

    argparse.add_argument('--task', type=str, default= "char_reversal")
    argparse.add_argument('--data', type=str, default= None)
    argparse.add_argument('--device', type=str, default='cuda')
    argparse.add_argument('--logdir', type=str, default='logs')

    args = argparse.parse_args()
    
    if (args.task == "char_reversal"):

        n_epochs = 10
        iter_per_epoch = 100        
        batch_size = 32
        learning_rate = 0.001
        block_length = 10
        channel_size = 256

        gen = CharacterReversalDatagen(length=block_length)

        if args.data is None:
            df = gen.generate_data(1e5)
        
        else:
            data_file = input("Enter the data file to upload: ")
            df = pd.read_csv(data_file)

        df = gen.encode_data(df)
        train_df, test_df = split_data(df, test_size= 0.2) # (N x L)
        train_x, train_y = to_tensor(train_df)
        test_x, test_y = to_tensor(test_df)

    elif (args.task == "shakespeare"):
        raise NotImplementedError("Shakespeare task not implemented")
    
    else:
        raise ValueError("Invalid task")

