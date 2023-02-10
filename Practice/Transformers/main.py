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

    def __init__(self):

        super().__init__()
        

    def forward(self, x, y = None):
        raise NotImplementedError

    def generate(self):
        raise NotImplementedError

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
    argparse.add_argument('--epochs', type=int, default=100)
    argparse.add_argument('--batch_size', type=int, default=32)
    argparse.add_argument('--learning_rate', type=float, default=0.001)
    argparse.add_argument('--embedding_size', type=int, default=256)
    argparse.add_argument('--block_length', type=int, default=10)

    args = argparse.parse_args()
    
    if (args.task == "char_reversal"):

        n_epochs = args.epochs
        batch_size = args.batch_size
        learning_rate = args.learning_rate
        block_length = args.block_length
        embedding_size = args.embedding_size

        gen = CharacterReversalDatagen(length=block_length)

        if args.data is None:

            df = gen.generate_data(1e5)
            df = gen.encode_data(df)
            train_df, test_df = split_data(df, test_size= 0.2) # (N x L)
        
        else:

            data_file = input("Enter the data file to upload: ")
            df = pd.read_csv(data_file)
            df = gen.encode_data(df)
            train_df, test_df = split_data(df, test_size= 0.2)

    elif (args.task == "shakespeare"):
        raise NotImplementedError("Shakespeare task not implemented")
    
    else:
        raise ValueError("Invalid task")




