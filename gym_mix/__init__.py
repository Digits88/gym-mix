from gym.envs.registration import register
from . import envs

register(
    id='DoubleLink-v0',
    entry_point='gym_mix.envs:DoubleLinkEnv',
    timestep_limit=100, # TODO
    reward_threshold=100,
    )

register(
    id='ContinuousCopy-v0',
    entry_point='gym_mix.envs:ContinuousCopyEnv',
    reward_threshold=1,
    )

register(
    id='ContinuousCopyRand-v0',
    entry_point='gym_mix.envs:ContinuousCopyRandEnv',
    reward_threshold=1,
    )

register(
    id='Chain-v0',
    entry_point='gym_mix.envs:ChainEnv',
    reward_threshold=10,
    )

register(
    id='PuddleWorld-v0',
    entry_point='gym_mix.envs:PuddleWorldEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 150},
    reward_threshold=10000, # TODO
    )

register(
    id='CollectCoin-v0',
    entry_point='gym_mix.envs:CollectCoinEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    reward_threshold=10000, # TODO
    )