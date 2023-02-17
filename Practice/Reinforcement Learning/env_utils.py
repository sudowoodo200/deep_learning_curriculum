## Basic imports
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.distributions import Categorical
import numpy as np, pandas as pd, os, sys, argparse
from datetime import datetime as dt
import matplotlib.pyplot as plt, seaborn as sns
from typing import Type, List, Callable, Dict, AnyStr

## Importing the RL environment engine
import gymnasium as gym

## Supporting class for storing training rollout trajectories
class RolloutStorage():

    def __init__(self, n_steps: int, num_envs: int, envs:Type[gym.vector.SyncVectorEnv], **kwargs):

        ## Parameters
        self.num_envs = num_envs
        self.n_steps = n_steps

        ## Training storage
        self.state = torch.zeros((self.n_steps, self.num_envs) + envs.single_observation_space.shape, dtype=torch.float)
        self.action = torch.zeros((self.n_steps, self.num_envs) + envs.single_action_space.shape, dtype=torch.long)
        self.reward = torch.zeros((self.n_steps, self.num_envs), dtype=torch.float)
        self.done = torch.zeros((self.n_steps, self.num_envs), dtype=torch.long)
        self.logprob = torch.zeros((self.n_steps, self.num_envs), dtype=torch.float)
        self.value = torch.zeros((self.n_steps, self.num_envs), dtype=torch.float)

        ## Advantages
        self.advantages = None

    def reset(self):
            
        self.state = torch.zeros_like(self.state)
        self.action = torch.zeros_like(self.action)
        self.reward = torch.zeros_like(self.reward)
        self.done = torch.zeros_like(self.done)
        self.logprob = torch.zeros_like(self.logprob)
        self.value = torch.zeros_like(self.value)
    
    def update(self, step, state, action, reward, done, logprob, value):

        self.state[step] = state        # (num_env x (state_dim) )
        self.action[step] = action      # (num_env x (action_dim) )
        self.reward[step] = reward      # (num_env x 1 )
        self.done[step] = done          # (num_env x 1 )
        self.logprob[step] = logprob    # (num_env x 1 )
        self.value[step] = value        # (num_env x 1 )
    
    def compute_gae(self, next_value: Type[torch.Tensor], next_done: Type[torch.Tensor], gamma:float, l:float,  **kwargs):

        ## Initialize the next state variables
        ## TODO: Understand why you want to re-initialize the advantage estimator
        gae = torch.zeros((self.num_envs), dtype=torch.float)
        self.advantages = torch.zeros((self.n_steps, self.num_envs), dtype=torch.float)

        ## Compute the advantages
        for step in reversed(range(self.n_steps)):
            
            ## Compute the delta and advantage
            delta = self.reward[step] + gamma * next_value * (1 - next_done) - self.value[step]
            gae = delta + gamma * l * (1 - next_done) * gae
            self.advantages[step] = gae

            ## Update the next state variables
            next_done = self.done[step]
            next_value = self.value[step]


## Main environment controller class
class EnvController:

    def __init__(self, env_name: str, n_steps: int, num_envs: int,  seed: int = 1, video_debug = False, run_name = "test", **kwargs):

        self.n_steps = n_steps
        self.num_envs = num_envs
        self.envs = gym.vector.SyncVectorEnv( 
            [ self.make_env(env_name, (seed + i), i, video_debug, run_name) for i in range(self.num_envs) ]
        )

        ## Error checks
        assert(isinstance(self.envs.single_action_space, gym.spaces.Discrete))  ## Discrete action spaces only for now

        ## Get the state and action dimensions
        self.state_space_shape = self.envs.single_observation_space.shape
        self.action_space_shape = self.envs.single_action_space.shape
        self.state_dim = np.array(self.state_space_shape).prod()
        self.action_dim = self.envs.single_action_space.n

        ## Training storage
        self.rollout_storage = RolloutStorage(self.n_steps, self.num_envs, self.envs)

    ## Do one n_step rollout for each environment instance in parallel
    def rollout(self, agent:Type[Callable], init_state:Type[torch.Tensor] = None, **kwargs):

        ## Reset the environment
        done = torch.ones(self.num_envs, dtype=torch.long)
        if init_state is None:
            state = torch.Tensor(self.envs.reset()[0])
        else:
            state = init_state

        ## Run the simulation
        for step in range(self.n_steps):

            ## Get and take the next action
            with torch.no_grad():
                action, logprob, entropy, value = agent(state)
                next_state, reward, next_terminated, next_truncated, info = self.envs.step(action.numpy())
                next_done = next_terminated | next_truncated

            ## Update the trajectory
            self.rollout_storage.update(step, state, action, torch.Tensor(reward).view(-1), done, logprob, value.flatten())

            ## Update the state
            state = torch.Tensor(next_state)
            done = torch.Tensor(next_done)

            ## TODO: How do you handle when one of the environments is done?
            ##       Do you just continue? Or do you stop the rollout?
            ##       For now, I'm just going to continue.

            ## TODO: Tensorboard logging of info
        
        return state, done  ## Return the final state / done variable for the next rollout

    ## General Advantage Estimation
    def gae(self, next_value: Type[torch.Tensor], next_done: Type[torch.Tensor], gamma = 0.99, l = 0.95,  **kwargs):

        self.rollout_storage.compute_gae(next_value, next_done, gamma, l)

    ## Initialize the vector environment
    def make_env(self, gym_id:str, seed = 1, recorded_idx = 0, video_debug = False, run_name = "test"):

        def thunk():

            env = gym.make(gym_id, render_mode='rgb_array')
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if video_debug & (recorded_idx == 0):
                env = gym.wrappers.RecordVideo(env, f"logging_videos/{gym_id}/{run_name}")

            # for reproducibility 
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            
            return env

        return thunk

    # Render a single test environment
    def render_play(self, policy: Type[Callable], env_idx = 0, **kwargs):

        raise NotImplementedError

    ## Close
    def close(self):
        
        self.envs.close()


#################################################### Main, for testing ####################################################

if __name__ == "__main__":
    
    print("Testing utils.py...")
    envs = EnvController(env_name = "CartPole-v1", n_steps = 100, num_envs = 1, seed = 1, video_debug = False, run_name = "test")