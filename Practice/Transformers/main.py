import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd, os, json, requests as req, string, numpy as np, argparse

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_util import *
from train_util import *
from attention import *


## Model Definition
class DecoderModel(nn.Module):

    def __init__(self, vocab_size, channel_size, block_length, num_heads, dropout, n_layers=6):

        super().__init__()
        self.channel_embedding = nn.Embedding(vocab_size, channel_size)
        self.positional_encoding = nn.Embedding(block_length, channel_size)
        self.main_block = nn.Sequential(*[AttentionBlock(block_length, channel_size, num_heads, dropout) for _ in range(n_layers)])
        self.out = nn.Linear(channel_size, vocab_size)
        self.loss_fn = F.cross_entropy

    def forward(self, x, y = None):

        batch_size, block_length = x.shape
        token_embedding = self.channel_embedding(x)  # (batch_size, block_length, channel_size)
        positional_embedding = self.positional_encoding(torch.arange(block_length))  # (batch_size, block_length, channel_size)
        embedding = token_embedding + positional_embedding  # (batch_size, block_length, channel_size)
        
        # main block
        z = self.main_block(embedding) # (batch_size, block_length * channel_size)

        # output
        output = self.out(z) # (batch_size, block_length, vocab_size)

        # compute loss if applicable
        if y is not None:
            loss = self.compute_loss(output, y)
        else:
            loss = None

        return output, loss

    def generate(self, prompt, max_tokens = 100):

        output = prompt.clone()

        for i in range(max_tokens):

            logits, _ = self.forward(output)
            output = torch.cat([output, decode_logits(logits)], dim=-1)

        return output
        

    def compute_loss(self, output, y):

        batch_size, block_length, vocab_size = output.shape
        logits = output.view(-1, vocab_size) # (batch_size * block_length, vocab_size)
        target = y.view(-1) # (batch_size * block_length)
        loss = self.loss_fn(logits, target, ignore_index=-1, reduction="sum")

        return loss

## Training
def train(model, optimizer, train_x, train_y, batch_size):

    model.train()
    total_loss = 0
    N = len(train_x)

    for i in range(0, N-batch_size, batch_size):

        optimizer.zero_grad()

        xb, yb = get_batch(train_x, train_y, batch_size, i)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / N

## Evaluation
@torch.no_grad()
def evaluate(model, test_x, test_y):

    model.eval()
    N = len(test_x)

    logits, loss = model(test_x, test_y)

    return logits, loss.item() / N


## Main
if __name__ == '__main__':

    argparse = argparse.ArgumentParser()

    argparse.add_argument('--task', type=str, default= "char_reversal")
    argparse.add_argument('--data', type=str, default= None)
    argparse.add_argument('--device', type=str, default='cuda')
    argparse.add_argument('--logdir', type=str, default='logs')

    args = argparse.parse_args()
    
    if (args.task == "char_reversal"):

        # The goal of this exercise is to check that the masking works. Under the current setup, the model should be able to learn 
        # to reverse the order of the earlier characters in the string, but not the later ones. This is because the if the input is a
        # tuple (a, b, c) and the target is (x, y, z), then the masking prevents x from seeing c. This is a good sanity check to make sure
        # that the masking is working correctly.
        
        print("Initializing task: Character Reversal")

        n_epochs = 50
        batch_size = 32
        block_length = 5
        learning_rate = 0.01
        channel_size = 64
        n_head = 4
        n_layers = 1
        dropout = 0.1
        output_lookback = block_length

        cr = CharacterReversalData(max_length = block_length, min_length = block_length)
        vocab_size = cr.vocab_size
        decoder = cr.decode
        encoder = cr.encode

        if args.data is None:
            N = int(1e5)
            print(f"Generating synthetic data with n = {N} and vocab_size={vocab_size-1}...")
            df = cr.gen(N)
        
        else:

            if args.data == "raw":
                data_file = input("Enter the data file to upload (enter for data/char_reversal.csv): ", default="data/char_reversal.csv")
                df = pd.read_csv(data_file)
                df = encoder(df)
            
            elif args.data == "encoded":
                data_file = input("Enter the data file to upload (enter for data/char_reversal_embed.csv): ", default="data/char_reversal_embed.csv")
                df = pd.read_csv(data_file)
            
            else:
                raise ValueError("Invalid data type")

        train_df, test_df = split_data(df, test_size= 0.1) # (N x L)
        train_x, train_y = to_tensor(train_df)
        test_x, test_y = to_tensor(test_df)

    elif (args.task == "shakespeare"):
        
        print("Initializing task: Shakespeare")

        n_epochs = 50
        batch_size = 32
        block_length = 100
        learning_rate = 0.001
        channel_size = 256
        n_head = 4
        n_layers = 3
        dropout = 0.1
        output_lookback = 1

        sh = ShakespeareData(block_size=block_length)
        vocab_size = sh.vocab_size
        decoder = sh.decode
        encoder = sh.encode

        if args.data is None:
            N = int(1e5)
            print(f"Generating synthetic data with n = {N} and vocab_size={vocab_size-1}...")
            df = sh.gen(N)
        
        else:
            
            if args.data == "raw":
                data_file = input("Enter the data file to upload (enter for data/shakespeare.json): ") or "data/shakespeare.json"
                df = pd.read_json(data_file)
                df = encoder(df)
            
            elif args.data == "encoded":
                data_file = input("Enter the data file to upload (enter for data/shakespeare_embed.json): ") or "data/shakespeare_embed.json"
                df = pd.read_json(data_file)
            
            else:
                raise ValueError("Invalid data type")
            
            assert(len(df.x[0]) == block_length)

        train_df, test_df = split_data(df, test_size= 0.1) # (N x L)
        train_x, train_y = to_tensor(train_df)
        test_x, test_y = to_tensor(test_df)
    
    else:
        raise ValueError("Invalid task")

    # initialize model & optimizer
    print("Initializing model...")
    model = DecoderModel(vocab_size, channel_size, block_length, n_head, dropout, n_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train model
    print(f"Training on {n_epochs} epochs...")
    for i in range(n_epochs):

        train_loss = train(model, optimizer, train_x, train_y, batch_size)
        logits, test_loss = evaluate(model, test_x, test_y)
        output = decode_vocab(decode_logits(logits, lookback_block_size=output_lookback), decoder)
        sample = pd.concat(
                [pd.DataFrame(decode_vocab(test_x, decoder)), 
                    pd.DataFrame(decode_vocab(test_y, decoder)), 
                    pd.DataFrame(output)], 
            axis=1).sample(5)

        print(f"Epoch {i} | Training Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f} with sample output: \n {sample}")

    # save model
    print("Complete. Saving model...")
    model_file = input("Enter the model file to save (hit enter to cancel): ")
    if model_file != "":
        torch.save(model.state_dict(), model_file)