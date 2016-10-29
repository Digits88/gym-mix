from gym import core, spaces
import numpy as np

DEBUG = True



class CollectCoinEnv(core.Env):
  """
  Description text (note that CollectCoinEnv == pixelworld):

  In the pixelworld, shown on the left, the blue agent has to pick up the
  yellow coin and place it on the goal position while avoiding the moving
  black wall (the wall clock time is constant, the hole width is twice
  the size of the agent and it moves at half of the speed of the agent).
  The agent position is defined in [0, 1]**2, its action in [-0.05, 0.05]**2
  and the x-axis of the wall is fixed. Both the initial agent position and the
  center of the hole of the wall are drawn from an uniform distribution.
  The discount factor is gamma = 1 and at each time step the agent receives
  a penalty equal to its distance from the coin (if not collected yet) or from the goal (otherwise).
  Additionally, it receives a bonus of +1 for collecting the coin and for placing it on the goal, and  a penalty of -1 for bumping into the wall or into the environment boundaries. When interacting
  with the environment, the agent does not know its true state, but has access only to a 21 x 21 pixels
  representation of it. Using such a representation has two major consequences. First, being the
  true continuous space, the pixel discretization entails a loss of information, as it is not possible
  to distinguish between similar states. Second, some pixels, although relevant for representing the
  environment, are non-informative for the task. For instance, it is irrelevant to identify the whole wall,
  as knowing just the center of the hole is sufficient for a full representation of the true state.
  """

  metadata = {
    'render.modes': ['rgb_array'],
    'video.frames_per_second' : 15
  }

  goal = [0.9, 0.8];
  coin = [0.2, 0.1];
  # L2 distance = 0.989 -> 19.79 steps -> penalty of 0.5 * 20 * 0.9 = 8.5

  hole_y = 0.5
  hole_width = 0.2
  stepsize = 0.05
  step_hole = 0.01
  radius_agent = 0.1
  
  agentLB = np.zeros(2)
  agentUB = np.ones(2)

  stateLB = np.zeros(21*21)
  stateUB = np.ones(21*21)
  # stateLB = np.zeros(4)
  # stateUB = np.ones(4)

  actionLB = -np.ones(2)
  actionUB = np.ones(2)

  rewardLB = -1
  rewardUB = 1


  def __init__(self):
    
    self.observation_space = spaces.Box(self.stateLB, self.stateUB)
    self.action_space = spaces.Box(self.actionLB,self.actionUB)

    self.viewer = None

  def _reset(self):
    # corresponds to initstate

    bad = True
    while bad:
      agent = np.random.uniform(low=0, high=1, size=[2])
      bad = agent[1] > self.hole_y - self.radius_agent and agent[1] < self.hole_y + self.radius_agent

    self.agent = agent
    self.hole_center = np.random.uniform(low=0, high=1)
    self.has_coin = False
    
    return self.observe()

  def _step(self, action):
    agent = self.agent
    hole_center = self.hole_center
    has_coin = self.has_coin

    # Parse action

    af = 2. * action # scale factor realatively arbitrary

    a = af / np.max([np.linalg.norm(af),1])
    
    # Move agent
    next_agent = agent + a * self.stepsize;
    out_of_bounds = np.any(self.agentLB > next_agent) or np.any(self.agentUB < next_agent)
    crossing = (agent[1] < self.hole_y and next_agent[1] > self.hole_y) or \
      (agent[1] > self.hole_y and next_agent[1] < self.hole_y)

    blocked = crossing and not ( agent[0] > hole_center - self.hole_width/2. and \
      agent[0] < hole_center + self.hole_width/2. )

    next_agent = agent if blocked else np.clip(next_agent, self.agentLB, self.agentUB)
    
    # Move barrier
    next_hole_center = np.mod(hole_center + self.step_hole, 1); # TODO: correct mod?

    # Reward
    dist_coin = np.abs( np.linalg.norm(agent-self.coin))
    dist_goal = np.abs( np.linalg.norm(agent-self.goal))

    to_deliver = has_coin
    reward = 0.
    if to_deliver:
      reward = -dist_goal

    to_pick = not has_coin
    if to_pick:
      reward = -dist_coin

    # Check coin delivered
    terminal = False
    delivered = has_coin and np.linalg.norm(next_agent-self.goal) < self.radius_agent
    if delivered:
      terminal = True
      reward = 1.
    
    # Check coin picked
    next_has_coin = has_coin;

    if np.linalg.norm(next_agent-self.coin) < self.radius_agent:
      next_has_coin = True

    if next_has_coin and not has_coin:
      reward = 1.
    
    # Penalties
    if out_of_bounds or blocked:
      reward = reward-1

    # Create next state
    self.agent = next_agent
    self.hole_center = next_hole_center
    self.has_coin = next_has_coin

    reward = 1. * reward # TODO: set scale factor for rewards correctly

    return (self.observe(), reward, terminal, {})


  def observe(self):
    pixels = self.features()
    f = np.reshape(pixels, -1) / 20. # TODO: set scale factor for observation correctly
    # f = np.concatenate([self.agent, [self.hole_center], [self.has_coin]])
    return f

  def _render(self, mode='human', close=False):
    pixels = self.features(rgb = True)
    return pixels



  def features(self,rgb = False):

    def sc(x,wideclip=False):
      c = 1 if wideclip else 0
      return tuple(np.clip(np.round(np.array(x)*[20.,-20.]+[0,20]).astype('int'),[0-c,0-c],[20+c,20+c])) # tuple to trigger normal indexing

    if rgb:
      pixels = np.zeros([21,21,3],dtype=np.uint8)
      wall_value = [255,255,255]
      agent_value = [0,0,255]
      goal_value = [0,255,0]
      coin_value = [255,255,0]
    else:
      pixels = np.zeros([21,21])
      wall_value = -10
      agent_value = -5
      goal_value = 10
      coin_value = 20

    # coin and goal pixel
    pixels[sc(self.goal)+(None,)] = goal_value

    if not self.has_coin:
      pixels[sc(self.coin)+(None,)] = coin_value

    # agent
    def cl(x):
      return np.clip(x,0,20)

    c = sc(self.agent)
    pixels[c[0], c[1], ...] = agent_value
    pixels[cl(c[0]+1), c[1], ...] = agent_value
    pixels[cl(c[0]-1), c[1], ...] = agent_value
    pixels[c[0], cl(c[1]+1), ...] = agent_value
    pixels[c[0], cl(c[1]-1), ...] = agent_value

    # wall left part
    ul1 = sc([0, self.hole_y],wideclip=True)
    br1 = sc([self.hole_center - self.hole_width / 2., self.hole_y],wideclip=True)
    pixels[ul1[0] : br1[0]+1, ul1[1] : br1[1]+1, ...] = wall_value

    # wall right part
    ul2 = sc([self.hole_center + self.hole_width / 2. , self.hole_y],wideclip=True)
    br2 = sc([1, self.hole_y],wideclip=True)
    pixels[ul2[0] : br2[0]+1, ul2[1] : br2[1]+1, ...] = wall_value

    pixels = pixels.T

    if rgb:
      import scipy.misc
      p = scipy.misc.imresize(pixels,4.,interp='nearest')
      return p 
    else:
      return pixels




