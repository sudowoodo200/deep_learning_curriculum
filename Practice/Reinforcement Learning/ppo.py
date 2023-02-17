## Baby PPO implementation
## Original PPO paper: https://arxiv.org/pdf/1707.06347.pdf
## Pieter Lecture (19:50): https://www.youtube.com/watch?v=KjWF8VIMGiY&ab_channel=PieterAbbeel
## Good W&B Resource: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ 

## Basic imports
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np, pandas as pd, os, sys, argparse
from datetime import datetime as dt
import matplotlib.pyplot as plt, seaborn as sns
from typing import Type, List
from distutils.util import strtobool

## Import tensorboard
from torch.utils.tensorboard import SummaryWriter

## Proprietary Packages
import env_utils as utl
import ppo_agent as agt

class PPO:

    def __init__(self, envs: Type[utl.EnvController], agent: Type[agt.PPOAgent], \
            gamma=0.99, l=0.95, n_epochs = 10, n_batch = 4, norm_adv = True, lr = 1e-3, lr_anneal = True,
            ratio_clip = 0.2, value_loss_clip = 0.5, policy_loss_coef = 1, value_loss_coef = 0.5, entropy_loss_coef = 0.01,
            grad_norm_clip = 0.5, device="cpu", **kwargs ):
        
        self.envs = envs
        self.agent = agent
        self.gae_params = {"gamma": gamma, "l": l}
        self.training_param = {"n_epochs": n_epochs, "n_batch": n_batch, "norm_adv": norm_adv, "lr": lr , "lr_anneal": lr_anneal,
                                "ratio_clip": ratio_clip, "value_loss_clip": value_loss_clip, "policy_loss_coef": policy_loss_coef, 
                                "value_loss_coef": value_loss_coef, "entropy_loss_coef": entropy_loss_coef, "grad_norm_clip": grad_norm_clip}
        self.device = device

    def train(self, n_iter = 100, **kwargs):

        ## Training Loop
        for i in range(n_iter):

            next_state, next_done = self.envs.rollout(self.agent)
            with torch.no_grad():
                next_value = self.agent.get_value(next_state).flatten()
            self.envs.gae(next_value, next_done, **self.gae_params)
            
            ## Learning Rate Annealing
            if self.training_param["lr_anneal"]:
                frac = 1.0 - ((i+1) / n_iter)
                self.training_param["lr"] = self.training_param["lr"] * frac

            ## Update agent
            training_loss = self.agent.update(self.envs.rollout_storage, **self.training_param)
            # print(f"Training Loss: {training_loss}")

        return self.agent.get_action, self.agent.get_value, training_loss


#################################################### Main ####################################################
if __name__ == "__main__":

    ## Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to run in parallel")
    parser.add_argument("--n_steps", type=int, default=200, help="Number of steps to run in each environment per iteration")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--video_debug", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Whether to record videos of the environment")
    parser.add_argument("--run_name", type=str, default="test", help="Name of the run")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    args = parser.parse_args()

    ## Initialize Tensorboard
    log_path = args.log_dir + "/" + args.run_name
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    ## Initialize the vector environment & agent, wrap in PPO
    envs = utl.EnvController(args.env, args.n_steps, args.num_envs, args.seed, args.video_debug)
    agent = agt.PPOAgent(envs)
    ppo = PPO(envs, agent)

    ## Train
    ppo.train(n_iter = int(1e4))

    ## Close Environments
    envs.close()

    ## Render a test environment with agent
    ## envs.render_play(ppo.agent.get_action)
