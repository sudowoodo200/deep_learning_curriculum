import torch, torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import os, json, time, sys, argparse

class MNISTData:

    def __init__(self, data = None, transform = None):

        # Load data with transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            self.transform = transform

        if data is None:
            self.train_data = datasets.MNIST(root='MNIST_data/', download=True, train=True, transform=self.transform)
            self.val_data = datasets.MNIST(root='MNIST_data/', download=True, train=False, transform=self.transform)
        else:
            raise NotImplementedError
        
        self.n_train = len(self.train_data)
        self.n_val = len(self.val_data)
        self.input_dim = self.train_data[0][0].size

    def get_dataloader(self, train = True, batch_size = 32, shuffle = True, perc_use = 1.0):
        data = self.train_data if train else self.val_data
        data = torch.utils.data.Subset(data, np.random.choice(len(data), int(len(data)*perc_use)))
        return DataLoader(data, batch_size = batch_size, shuffle = shuffle)

def contrastive_loss_fn(y_pred, y_true, epsilon = 1.0, reduction = "sum"):
    
    # breakpoint()

    N = len(y_true)
    y_true = y_true.float() ## (N, C)
    y_pred = y_pred.float() ## (N)

    y_diff = torch.stack( [torch.roll(y_pred, shifts=i, dims=0) - y_pred for i in range(N)] , dim = 0) ## (N, N, C)
    y_l2_diff = (torch.abs(y_diff)**2).sum(dim = -1)  ## (N, N)
    y_inverse_l2_diff = torch.clamp(epsilon - y_l2_diff, min = 0.0)**2 ## (N, N)
    mask = (y_true.unsqueeze(0) == y_true.unsqueeze(1)).type(torch.int64) ## (N, N)

    loss = (mask * y_l2_diff + (1 - mask) * y_inverse_l2_diff).sum(dim=1).sum(dim=0).float() / 2 ## (N, N)

    if reduction == "sum":
        pass
    elif reduction == "mean":
        loss = loss / N
    else:
        raise NotImplementedError
    
    return loss


############ Main ############

if __name__ == '__main__':

    # Load data
    data = MNISTData()

    # Get dataloader
    train_loader = data.get_dataloader(train = True, batch_size = 32, shuffle = True)
    val_loader = data.get_dataloader(train = False, batch_size = 32, shuffle = True)
    breakpoint()

    # Show some images
    for i, (images, labels) in enumerate(train_loader):
        if i == 0:
            plt.figure(figsize = (10, 10))
            for j in range(4):
                plt.subplot(1, 4, j + 1)
                plt.imshow(images[j].reshape(28, 28), cmap = 'gray')
                plt.title(labels[j].item())
            plt.show()
            break