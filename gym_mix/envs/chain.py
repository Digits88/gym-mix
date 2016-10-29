from gym import core, spaces
import numpy as np

DEBUG = True

class ChainEnv(core.Env):

  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second' : 15
  }

  shape = [100]

  def __init__(self):
    self.n = 3 # max n = 49
    self.observation_space = spaces.Box(np.zeros(self.shape),np.ones(self.shape))
    self.action_space = spaces.Discrete(2)
    self.viewer = None
    self.obs = np.zeros([100])

  def _reset(self):
    self.t = 1
    self.state = self.n
    return self.observe()

  def _step(self, action):
    # action in {0,1}
    self.state = np.clip(self.state - 1 + 2 * action,0,2*self.n)
    
    if self.state == 0:
      reward = .001
    elif self.state == 2 * self.n:
      reward = 1
    else:
      reward = 0

    terminal = not self.t < 9 + self.n

    self.t += 1

    return (self.observe(), reward, terminal, {})


  def observe(self):
    self.obs[:] = 0
    self.obs[:self.state] = 1
    pixels = np.reshape(self.obs, self.shape)
    return pixels

