from gym import core, spaces
import numpy as np

DEBUG = True

class ChainEnv(core.Env):

  metadata = {
    'render.modes': [],
    'video.frames_per_second' : 15
  }

  max_return = 10

  def __init__(self, n = 4):
    self.n = n  # max n = 49
    self.observation_space = spaces.Box(np.zeros(self.n+2), np.ones(self.n+2))
    self.action_space = spaces.Discrete(2)

  def _reset(self):
    self.t = 0
    self.state = 1
    return self.features()

  def _step(self, action):
    self.state = int(np.clip(self.state + np.squeeze(action) * 2 - 1, 0, self.n+1))

    r = .0001 * (self.state == 0) + 1. * (self.state == self.n+1)
    self.t += 1

    terminal = self.t >= 9 + self.n
    return self.features(), r, terminal, {}

  def features(self):
    f = np.zeros(self.n+2)
    f[self.state] = 1
    return f


class ContinuousChainEnv():

  metadata = {
    'render.modes': [],
    'video.frames_per_second' : 15
  }

  max_return = 10

  def __init__(self, n = 4):
    self.n = n  # max n = 49
    self.observation_space = spaces.Box(np.zeros(self.n), np.ones(self.n))
    self.action_space = spaces.Box(np.array([-1]), np.array([1]))


  def _step(self, action):
    action = np.clip(np.squeeze(action), -1, 1)

    self.state = np.clip(self.state + action, 0, self.n + 1)

    pos = np.atleast_2d(np.arange(0, self.n)) - self.state
    features = np.exp(- np.square(pos))

    reward = .0001 * (self.state < 1) + 1. * (self.state > self.n)
    terminal = self.t >= 9 + self.n

    return features, reward, terminal, {}