## Small convolutional neural network for character recognition using pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import os, json, time, sys, argparse, itertools
from datetime import datetime as dt

import tensorboard
from tensorboardX import SummaryWriter
import tqdm

from utils import *

class CNN(nn.Module):

    def __init__(self, input_channel = 1, internal_channel = 4, output_channel = 2, conv_kernel_size = 8, stride = 1, padding = 2, input_dim = 28, dropout = 0.2):
        
        super().__init__()
        self.input_dim = input_dim
        self.input_channel = input_channel
        self.output_channel = output_channel

        self.convolution_block_1 = ConvolutionModule(input_channel, internal_channel, conv_kernel_size, stride, padding, self.input_dim)
        self.conv_block_1_output_dimension = self.convolution_block_1.output_dim
        print(f"Conv block 1 initialized with output dimension: {self.conv_block_1_output_dimension}")

        self.convolution_block_2 = ConvolutionModule(internal_channel, internal_channel, conv_kernel_size, stride, padding, self.conv_block_1_output_dimension)
        self.conv_block_2_output_dimension = self.convolution_block_2.output_dim
        print(f"Conv block 2 initialized with output dimension: {self.conv_block_2_output_dimension}")

        print(f"Initializing Linear layer with input dimension: {int(self.convolution_block_2.output_channel * self.conv_block_2_output_dimension**2)}")
        self.dropout_1 = nn.Dropout(dropout)
        self.ff_1 = nn.Linear(int(self.convolution_block_2.output_channel * self.conv_block_2_output_dimension**2), 128)

        print(f"Initializing Linear layer with input dimension: {128}")
        self.dropout_2 = nn.Dropout(dropout)
        self.ff_2 = nn.Linear(128, output_channel)

        self.logits = nn.LogSoftmax(dim = -1)
        self.loss_fn = F.cross_entropy

        self._initialize_weights(mean = 0, std = 0.05)
    
    def _initialize_weights(self, mean, std):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def forward(self, x, y = None):
        
        x = self.convolution_block_1(x)
        x = self.convolution_block_2(x)
        x = x.view(-1, int(self.convolution_block_2.output_channel * self.conv_block_2_output_dimension**2))
        x = self.dropout_1(x)
        x = self.ff_1(x)
        x = self.dropout_2(x)
        x = self.ff_2(x)
        x = self.logits(x)

        if y is not None:
            loss = self.compute_loss(x, y)
        else:
            loss = None
        
        return x, loss

    def compute_loss(self, x, y):
        
        logits = x.view(-1, self.output_channel)
        labels = y.view(-1)
        loss = self.loss_fn(logits, labels, reduction="sum")
        return loss

class ConvolutionModule(nn.Module):

    def __init__(self, input_channel, output_channel, conv_kernel_size, stride, padding, input_dim, pool_size = 2):

        super().__init__()

        self.conv = nn.Conv2d(input_channel, output_channel, conv_kernel_size, stride, padding)
        self.maxpool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        self.activation = nn.ReLU()

        self.input_dim = input_dim
        self.input_channel = input_channel
        self.conv_kernel_size = conv_kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_size = pool_size
        self.output_channel = output_channel
        self.output_dim = self.get_output_dim()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x
    
    # assume square image
    def get_output_dim(self):
        conv_output_dim = (self.input_dim - self.conv_kernel_size + 2 *self.padding) / self.stride + 1
        activation_dim = conv_output_dim
        maxpool_dim = (activation_dim - self.pool_size)/self.pool_size + 1
        return int(maxpool_dim)


############################################################################################################
# Path: Practice/Basic Character Recognition/train.py
## Training script for character recognition using pytorch

# Training module
def train(model, optimizer, train_data, batch_size, learning_rate, log_dir = None, model_dir = None):
    
    model.train()
    total_train_loss = 0
    for images, labels in train_data:
        optimizer.zero_grad()
        logits, loss = model(images, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    return total_train_loss / len(train_data)

# Testing module
def test(model, test_data, log_dir = None, model_dir = None):
    
    model.eval()
    total_test_loss = 0
    for images, labels in test_data:

        logits, loss = model(images, labels)
        total_test_loss += loss.item()
    
    return total_test_loss / len(test_data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=f"logs/CNN/{dt.now().strftime('%Y%m%d-%H%M%S')}")

    args = parser.parse_args()

    ## Model parameters
    input_channel = 1
    internal_channel = 32 ## range (10, 50, 10)
    output_channel = 10
    conv_kernel_size = 4
    stride = 1
    padding = 1
    input_dim = 28
    dropout = 0.2

    ## initialize model
    print("Initializing model...")
    model = CNN(input_channel, internal_channel, output_channel, conv_kernel_size, stride, padding, input_dim, dropout)
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ## Training parameters
    n_epoch = 10
    batch_size = 32 
    learning_rate = 1e-3 ## Prop to 1/sqrt(model_size)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    ## Load data
    print("Loading data...")
    data = MNISTData()
    percent_used = 0.5 ## parameter: range (0.2, 1, 0.2)
    train_loader = data.get_dataloader(train = True, batch_size = batch_size, shuffle = True, perc_use=percent_used)
    val_loader = data.get_dataloader(train = False, batch_size = batch_size, shuffle = True)

    # Global Logging with Tensorboard
    """ writer = SummaryWriter(args.log_dir) """

    ## Training Model
    print(f"Training with {model_size} parameters and {percent_used*100}% of the data")
    for i in range(n_epoch):

        trg_loss = train(model, optimizer, train_loader, batch_size, learning_rate, args.log_dir, args.model_dir)
        val_loss = test(model, val_loader, args.log_dir, args.model_dir)
        print(f"Epoch {i+1}: Training loss: {trg_loss}, Validation loss: {val_loss}")


    ## Save Model
    if args.model_dir is not None:
        print(f"Saving model to {args.model_dir}...")
        torch.save(model.state_dict(), args.model_dir + "/" + "char_cnn.pt")

