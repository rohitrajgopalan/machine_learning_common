# --------------------------------------
# Ornstein-Uhlenbeck Noise
# Author: Flood Sung and Max Ferg
# Date: 2016.5.4
# Reference 1: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
# Reference 2: https://github.com/maxkferg/DDPG/blob/master/ddpg/ou_noise.py
# --------------------------------------

import numpy as np
import numpy.random as nr


class OUNoise:
    def __init__(self, action_dimension, mu, theta, sigma):
        self.action_dimension = action_dimension
        self.mu = np.full(self.action_dimension, mu) if type(mu) in [int, float] else mu
        self.theta = theta
        self.sigma = np.full(self.action_dimension, sigma) if type(sigma) in [int, float] else sigma
        self.state = self.mu
        self.reset()

    def reset(self):
        self.state = self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        return x + dx
