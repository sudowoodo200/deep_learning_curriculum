## Good Resources for Adam Optimizer and the Pytorch Optimizers library
## https://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html
## https://arxiv.org/pdf/1412.6980.pdf (Adam paper)

## Unclear how to use / install Triton

import torch
from torch.optim.optimizer import Optimizer, required

class Adam(Optimizer):

    def __init__(self, params, lr=required, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        self.params = params
        self.defaults = dict(lr=required, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        super().__init__(self.params, self.defaults)
    
