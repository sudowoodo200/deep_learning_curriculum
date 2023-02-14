import numpy as np
import matplotlib.pyplot as plt
from pydantic.dataclasses import dataclass
from typing import List, Tuple, Dict, Union, Optional, Any

class DummyModel:

    def _init__(self, init_theta: Optional[np.ndarray] = None):

        self.dim = 1e3
        self.H = np.diag([1/x for x in range(1,self.dim+1)])
        self.mu = np.zeros(self.dim)
        self.C = np.diag([1/x for x in range(1,self.dim+1)])

        if init_theta is None:
            self.theta = np.random.normal(0,1, self.dim)
        else:
            self.theta = init_theta

    def L(self) -> float:
        return 0.5 * np.dot(np.dot(self.theta.T, self.H), self.theta)
    
    def g(self) -> np.ndarray:
        epsilon = np.random.normal(self.mu, self.C)
        return np.dot(self.H, self.theta) + epsilon
    
class DummyOptimizer:

    def __init__(self, model: DummyModel, batch_size: int, max_steps: int = 1000):
        
        self.steps = 0
        self.model = model
        self.B = batch_size
        self.step_size = 1e-3
        self.max_steps = max_steps

        # Initialize DP
        self.thetas = np.zeros(self.max_steps, self.model.dim)
        self.mean_thetas = np.zeros(self.max_steps, self.model.dim)
        self.var_thetas = np.zeroes(self.max_steps, self.model.dim)

        self.thetas[0] = self.model.theta
        self.mean_thetas[0] = self.thetas[0]
        self.var_thetas[0] = np.ones(self.model.dim)

class DummySGD(DummyOptimizer):

    raise NotImplementedError

class DummyAdam(DummyOptimizer):

    raise NotImplementedError