from gym import core, spaces
import numpy as np

DEBUG = True

class PuddleWorldEnv(core.Env):

  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second' : 15
  }

  goal = [1.,1.]
  stepsize = .05

  p1 = [[.1, .75],[.45, .75]]  # Centers of the first puddle
  p2 = [[.45, .4],[.45, .8]]  # Centers of the second puddle
  p3 = [[.8, .2],[.8, .5]]  # Centers of the third puddle
  p4 = [[.7, .75],[.7, .8]]  # Centers of the fourth puddle

  radius = .1

  n_centers = 10. # rbf centers

  stateLB = np.zeros(n_centers*n_centers)
  stateUB = np.ones(n_centers*n_centers)

  actionLB = - np.array([1,1])
  actionUB = np.array([1,1])

  rewardLB = -41
  rewardUB = -1


  def __init__(self):
    self.observation_space = spaces.Box(self.stateLB, self.stateUB)
    self.action_space = spaces.Box(self.actionLB,self.actionUB)

    self.viewer = None

    # rbfs
    a = np.linspace(0, 1, self.n_centers)
    b, c = np.meshgrid(a,a)
    self.rbf_centers = np.concatenate([b.reshape([-1,1]), c.reshape([-1,1])],axis=1)
    # print(self.rbf_centers)


  def _reset(self):
    self.state = np.random.uniform(low=0, high=1, size=[2])

    return self.observe()

  def _step(self, action):
    a2 = action / np.linalg.norm(action) * self.stepsize # TODO: normalizing correctly?

    noise = np.random.multivariate_normal([0,0],[[.0001, 0],[0, .0001]])
    s2 = self.state + a2 + noise
    self.state = np.clip(s2, [0,0], [1,1])

    reward = 1. * (self.puddlepenalty() - 1)
    terminal = np.sum(self.state) >= 1.9

    return (self.observe(), reward, terminal, {})


  def observe(self):
    r = 1. 
    phi = np.exp( - self.n_centers**2 / r**2 * np.sum((self.state - self.rbf_centers)**2, axis=1))
    #phi2 = np.squeeze(np.reshape(phi, -1))
    return phi


  def puddlepenalty(self):
    factor = 400

    if self.state[0] > self.p1[1][0]:
      d1 = np.linalg.norm(np.transpose(self.state) - self.p1[1])
    elif self.state[0] < self.p1[0][0]:
      d1 = np.linalg.norm(np.transpose(self.state) - self.p1[0])
    else:
      d1 = np.abs(self.state[1] - self.p1[0][1])

    if self.state[1] > self.p2[1][1]:
      d2 = np.linalg.norm(np.transpose(self.state) - self.p2[1])
    elif self.state[1] < self.p2[0][1]:
      d2 = np.linalg.norm(np.transpose(self.state) - self.p2[0])
    else:
      d2 = np.abs(self.state[0] - self.p2[0][0])

    if self.state[1] > self.p3[1][1]:
      d3 = np.linalg.norm(np.transpose(self.state) - self.p3[1])
    elif self.state[1] < self.p3[0][1]:
      d3 = np.linalg.norm(np.transpose(self.state) - self.p3[0])
    else:
      d3 = np.abs(self.state[0] - self.p3[0][0])

    if self.state[1] > self.p4[1][1]:
      d4 = np.linalg.norm(np.transpose(self.state) - self.p4[1])
    elif self.state[1] < self.p4[0][1]:
      d4 = np.linalg.norm(np.transpose(self.state) - self.p4[0])
    else:
      d4 = np.abs(self.state[0] - self.p4[0][0])

    min_distance_from_puddle = np.min([d1, d2, d3, d4])

    if min_distance_from_puddle <= self.radius:
      reward = - factor * (self.radius - min_distance_from_puddle)
    else:
      reward = 0

    return reward

  def _render(self, mode='human', close=False):

    from gym.envs.classic_control import rendering
    if close:
      if self.viewer is not None:
        self.viewer.close()
      return
    
    if self.viewer is None:
      self.viewer = rendering.Viewer(512,512) if mode=='human' else rendering.Viewer(512,512)
      self.viewer.set_bounds(0,1,0,1)

    circ = self.viewer.draw_circle(.02)
    circ.set_color(0,0, 0)
    jtransform = rendering.Transform(translation=self.state)
    circ.add_attr(jtransform)

    self.viewer.render()
    if mode == 'rgb_array':
      return self.viewer.get_array()
    elif mode is 'human':
      pass


if __name__ == '__main__':
  p = PuddleWorldEnv()
  o = p._reset()
