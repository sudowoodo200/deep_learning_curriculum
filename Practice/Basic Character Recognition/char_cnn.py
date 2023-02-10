## Small convolutional neural network for character recognition using pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, json, time, sys, argparse

import tensorboard
from tensorboard import SummaryWriter
import tqdm

from data_util import *

