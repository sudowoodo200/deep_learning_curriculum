## Basic imports
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.distributions import Categorical
import numpy as np, pandas as pd, os, sys, argparse
from datetime import datetime as dt
import matplotlib.pyplot as plt, seaborn as sns
from typing import Type, List, Callable, Dict, AnyStr

## Importing the RL environment engine
import gymnasium as gym
## import procgen as pg -- some unresolved pip install issues

## Importing proprietary modules
from env_utils import EnvController, RolloutStorage

## Main Agent Class
## TODO: Implement Actor and Critic separately to support different types of encoding / policy networks
class PPOAgent(nn.Module):

    def __init__(self, envs: Type[EnvController], optimizer = "adam", actor_params: Dict = None, 
                 critic_params: Dict = None, **kwargs):
        
        super().__init__()

        ## Get envs dimensions
        self.state_space_shape = envs.state_space_shape
        self.action_space_shape = envs.action_space_shape
        self.state_dim = envs.state_dim
        self.action_dim = envs.action_dim

        ## Set parameters
        self.actor_params = {"input_dim": self.state_dim, "hidden_dim": 64, "output_dim": self.action_dim} if actor_params is None else actor_params
        self.critic_params = {"input_dim": self.state_dim, "hidden_dim": 64, "output_dim": 1} if critic_params is None else critic_params
        
        ## Initialize the policy network
        self.policy = self.initialize_nn(**self.actor_params)
        self.value_function = self.initialize_nn(**self.critic_params)

        ## Initialize the advantage estimator
        self.advantages = None

        ## Initialize the optimizer
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def initialize_nn(self, input_dim: int, output_dim: int, hidden_dim: int = 64, **kwargs):

        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, state: Type[torch.Tensor], action: Type[torch.Tensor] = None, **kwargs):

        action, logprob, entropy = self.get_action(state, action)
        return action, logprob, entropy, self.get_value(state)
    
    def get_action(self, state: Type[torch.Tensor], action: Type[torch.Tensor] = None, **kwargs):

        logits = F.softmax(self.policy(state), dim=-1)
        dist = Categorical(logits=logits)
        action = dist.sample() if action is None else action
        logprob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, logprob, entropy
    
    def get_value(self, state: Type[torch.Tensor]):

        return self.value_function(state)
        
    ## Learning loop
    def update(self, rollout_storage: Type[RolloutStorage], n_epochs = 10, n_batch = 4, norm_adv = True, lr = 1e-3,
              ratio_clip = 0.2, value_loss_clip = 0.5, policy_loss_coef = 1, value_loss_coef = 0.5, entropy_loss_coef = 0.01,
               grad_norm_clip = 0.5, **kwargs):

        ## Update optimizer parameters
        self.optimizer.param_groups[0]["lr"] = lr

        ## Get and flatten the rollout data across all environment instances
        state = rollout_storage.state.reshape((-1,) + self.state_space_shape)       # (n_steps * num_envs, state_dim)
        action = rollout_storage.action.reshape((-1,) + self.action_space_shape)    # (n_steps * num_envs, action_dim)
        logprob = rollout_storage.logprob.reshape(-1)                               # (n_steps * num_envs)
        advantages = rollout_storage.advantages.reshape(-1)                         # (n_steps * num_envs)
        values = rollout_storage.value.reshape(-1)                                  # (n_steps * num_envs)
        returns = advantages + values

        ## Parameters
        N = rollout_storage.n_steps * rollout_storage.num_envs
        batch_size = N // n_batch
        idx = np.arange(N)

        ## Logging
        mean_loss = {"policy": 0, "value": 0, "entropy": 0, "total_loss": 0}

        # Optimizing the policy and value network
        for i in range(n_epochs):

            ## Shuffle the data
            np.random.shuffle(idx)

            ## Iterate over the batches
            for j in range(n_batch):

                ## Get the batch indices
                batch_idx = idx[j * batch_size : (j + 1) * batch_size]

                ## Get the batch data
                batch_state = state[batch_idx]                                      # (batch_size, state_dim)
                batch_action = action[batch_idx]                                    # (batch_size, action_dim)  
                batch_logprob = logprob[batch_idx]                                  # (batch_size)
                batch_advantages = advantages[batch_idx]                            # (batch_size)
                batch_values = values[batch_idx]                                    # (batch_size)
                batch_returns = returns[batch_idx]                                  # (batch_size)

                ## Get the policy loss
                _, new_logprob, entropy, new_value = self.forward(batch_state, batch_action)
                ratio = torch.exp(new_logprob - batch_logprob)

                if norm_adv:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-9)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - ratio_clip, 1.0 + ratio_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                ## Get the clipped value loss
                value_loss_unclipped = (new_value - batch_returns)**2
                value_loss_clipped = batch_values + torch.clamp(new_value - batch_values, value_loss_clip, value_loss_clip)
                value_loss_clipped = (value_loss_clipped - batch_returns)**2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                ## Get the entropy loss
                entropy_loss = entropy.mean()

                ## Get the total loss
                ## TODO: Study this part. Feels like arbitrary coefficients
                loss = policy_loss_coef * policy_loss + value_loss_coef * value_loss - entropy_loss_coef * entropy_loss
                
                ## Logging
                mean_loss["policy"] += policy_loss.item() / n_epochs                # Scalar
                mean_loss["value"] += value_loss.item() / n_epochs                  # Scalar
                mean_loss["entropy"] += entropy_loss.item() / n_epochs              # Scalar
                mean_loss["total_loss"] += loss.item() / n_epochs                   # Scalar

                ## Take a gradient step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_norm_clip)
                self.optimizer.step()
        
        return mean_loss
    



#################################################### Main, for testing ####################################################

if __name__ == "__main__":
    
    print("Testing utils.py...")
    envs = EnvController(env_name = "CartPole-v1", num_envs = 1, seed = 1, video_debug = True, run_name = "test")
    agent = PPOAgent(envs, state_dim = 4, action_dim = 2)