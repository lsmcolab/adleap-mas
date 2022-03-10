from gym.envs.registration import register

#####
# Major Environment
#####
register(
    id='AdhocReasoningEnv-v1',
    entry_point='src.envs:AdhocReasoningEnv',
)

#####
# Toy Problems Environments
#####
register(
    id='MazeEnv-v1',
    entry_point='src.envs:MazeEnv',
)

register(
    id='RockSampleEnv-v1',
    entry_point='src.envs:RockSampleEnv',
)
register(
    id='TigerEnv-v1',
    entry_point='src.envs:TigerEnv',
    )

#####
# Ad-hoc Teamwork Environment
#####
register(
    id='CaptureEnv-v1',
    entry_point='src.envs:CaptureEnv',
)

register(
    id='LevelForagingEnv-v1',
    entry_point='src.envs:LevelForagingEnv',
)

#####
# Realistic Scenarios
#####
register(
    id='SmartFireBrigadeEnv-v1',
    entry_point='src.envs:SmartFireBrigadeEnv',
)

#####
# Card Games
#####
register(
    id='TrucoEnv-v1',
    entry_point='src.envs:TrucoEnv',
)