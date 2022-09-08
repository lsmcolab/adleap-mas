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
    id='TigerEnv-v2',
    entry_point='src.envs:TigerEnv',
    )
register(
    id='MazeEnv-v2',
    entry_point='src.envs:MazeEnv',
)

register(
    id='RockSampleEnv-v2',
    entry_point='src.envs:RockSampleEnv',
)

#####
# Ad-hoc Teamwork Environment
#####
register(
    id='LevelForagingEnv-v2',
    entry_point='src.envs:LevelForagingEnv',
)
register(
    id='CaptureEnv-v2',
    entry_point='src.envs:CaptureEnv',
)


#####
# Realistic Scenarios
#####
register(
    id='SmartFireBrigadeEnv-v1',
    entry_point='src.envs:SmartFireBrigadeEnv',
)
register(
    id='TradeStockEnv-v1',
    entry_point='src.envs:TradeStockEnv',
)

#####
# Card Games
#####
register(
    id='TrucoEnv-v2',
    entry_point='src.envs:TrucoEnv',
)