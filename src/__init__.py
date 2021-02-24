from gym.envs.registration import registry, register, make, spec

register(
    id='AdhocReasoningEnv-v0',
    entry_point='src.envs:AdhocReasoningEnv',
)

register(
    id='LevelForagingEnv-v0',
    entry_point='src.envs:LevelForagingEnv',
)

register(
    id='TrucoEnv-v0',
    entry_point='src.envs:TrucoEnv',
)
