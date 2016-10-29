from gym.envs.registration import register

register(
    id='DoubleLink-v0',
    entry_point='gym_mix.envs:DoubleLinkEnv',
    timestep_limit=100, # TODO
    reward_threshold=100,
    )

register(
    id='ContinuousCopy-v0',
    entry_point='gym_mix.envs:ContinuousCopyEnv',
    timestep_limit=10000,
    reward_threshold=1,
    )

register(
    id='ContinuousCopyRand-v0',
    entry_point='gym_mix.envs:ContinuousCopyRandEnv',
    timestep_limit=10000,
    reward_threshold=1,
    )

register(
    id='Chain-v0',
    entry_point='gym_mix.envs:ChainEnv',
    timestep_limit=10000,
    reward_threshold=1,
    )

register(
    id='PuddleWorld-v0',
    entry_point='gym_mix.envs:PuddleWorldEnv',
    timestep_limit=150,
    reward_threshold=10000, # TODO
    )

register(
    id='CollectCoin-v0',
    entry_point='gym_mix.envs:CollectCoinEnv',
    timestep_limit=1000,
    reward_threshold=10000, # TODO
    )