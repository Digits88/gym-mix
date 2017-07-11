"""
Simple environment with known optimal policy and value function.

"""

import gym
import numpy as np
from gym import spaces

sc = np.array([.5])

class ContinuousCopyEnv(gym.Env):
    
    def __init__(self):
        self.action_space = spaces.Box(-sc,sc)
        self.observation_space = spaces.Box(-sc,sc)

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def next(self):
        self.state = np.array([.1])
        return self.state

    def _step(self, action):
        reward = np.squeeze(- 1. * np.abs(self.state-action))
        
        done = True
        return self.next(), reward, done, {}

    def _reset(self):
        return self.next()


class ContinuousCopyRandEnv(ContinuousCopyEnv):
    """
    Simple environment for agent debugging.
    The agent has to use the state to achieve an
    expected reward higher than -0.25.

    Dynamics:
        state ~ uniform(-0.5,0.5)
        reward = - abs(state-action)
    
    Known Policies:
        optimal policy a = s has E[r] = 0
        constant policy with a = 0 has E[r] = -0.25
        constant policies with a = -0.5 or 0.5 have E[r] = -0.5
        random policy with a ~ uniform(-0.5,0.5) has E[r] = 1/3 
    """
    def next(self):
        self.state = 1. * np.random.uniform(-sc,sc,size=[1])
        return self.state