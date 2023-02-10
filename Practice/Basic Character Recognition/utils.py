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
            self.train_data = datasets.MNIST(root='MNIST_data/', download=True, train=True, transform=transform)
            self.val_data = datasets.MNIST(root='MNIST_data/', download=True, train=False, transform=transform)
        else:
            raise NotImplementedError
        
        self.n_train = len(self.train_data)
        self.n_val = len(self.val_data)
        self.input_size = self.train_data[0][0].shape

    def get_dataloader(self, train = True, batch_size = 32, shuffle = True):
        data = self.train_data if train else self.val_data
        return DataLoader(data, batch_size = batch_size, shuffle = shuffle)


############ Main ############

if __name__ == '__main__':

    # Load data
    data = MNISTData()
    breakpoint()

    # Get dataloader
    train_loader = data.get_dataloader(train = True, batch_size = 32, shuffle = True)
    val_loader = data.get_dataloader(train = False, batch_size = 32, shuffle = True)

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