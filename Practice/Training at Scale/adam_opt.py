## Good Resources for Adam Optimizer and the Pytorch Optimizers library
## https://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html
## https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/optimization.py#L273 (HuggingFace implementation of AdamW)
## https://arxiv.org/pdf/1412.6980.pdf (Adam paper)

## Unclear how to use / install Triton. Doesn't support the x86_64 architecture on Mac.
## Undefined symbols for architecture arm64:
## "llvm::raw_ostream::SetBufferAndMode(char*, unsigned long, llvm::raw_ostream::BufferKind)", referenced from: _main in FileCheck.cpp.o

import torch
from torch.optim.optimizer import Optimizer, required

class JankyAdam(Optimizer):

    def __init__(self, params, lr=required, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        self.params = params
        self.defaults = {"lr": lr, "betas":betas, "eps":eps, "weight_decay":weight_decay}
        super().__init__(self.params, self.defaults)

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
    
    def step(self, closure=None):

        """
        Performs a single optimization step.
        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        ## param_groups is a list of dictionaries. Each dictionary contains the parameters and hyperparameters for a given group.
        ## Each group is a list of parameters. Each parameter is a tensor. This allows for different learning rates for different sets of parameters to be specified
        ## It also stores the hyperparameters for this group under different keys.
        ## See details on param_groups: https://pytorch.org/docs/stable/optim.html#per-parameter-options
        for p_group in self.param_groups:

            for p in p_group["params"]:

                ## tensor.grad contains the gradient of the tensor with respect to the loss function, if requires_grad is True
                ## If requires_grad is False or if the forward pass has not been done, tensor.grad is None
                ## See details on tensor.grad: https://pytorch.org/docs/stable/tensors.html#torch.Tensor.grad
                if p.grad is None:
                    continue
                grad = p.grad.data

                if grad.is_sparse:
                    raise NotImplementedError("Sparse gradients not supported")

                ## self.state is a dictionary that contains the variables of the optimizer for each parameter p
                ## for instance, the first and second moment of the gradient for each parameter in Adam.
                ## len(self.state[p]) == 0 if the optimizer has not been initialized for the parameter p
                ## See details on self.state: https://pytorch.org/docs/stable/optim.html#optimizer-state-dict
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                beta1, beta2 = p_group["betas"]
                state["step"] += 1

                ## Update the first and second moments, with decay
                state["exp_avg"] = state["exp_avg"] * beta1 + (1-beta1) * grad ## m_1[t] = beta1 * m_1[t-1] + (1 - beta1) * g(t)
                state["exp_avg_sq"] = state["exp_avg_sq"] * beta2 + (1-beta2) * grad ** 2 ## m_2[t] = beta2 * m_2[t-1] + (1 - beta2) * g(t)^2

                exp_avg_unbiased = state["exp_avg"] / (1 - beta1 ** state["step"]) ## m_1_hat[t] = m_1[t] / (1 - beta1^t)
                exp_avg_sq_unbiased = state["exp_avg_sq"] / (1 - beta2 ** state["step"]) ## m_2_hat[t] = m_2[t] / (1 - beta2^t)

                ## Update the parameters
                p.data = p.data - p_group["lr"] * exp_avg_unbiased / (torch.sqrt(exp_avg_sq_unbiased) + p_group["eps"])


        return loss