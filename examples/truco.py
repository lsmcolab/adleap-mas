###
# Imports
###
import sys
import os
sys.path.append(os.getcwd())
import time

from src.envs.TrucoEnv import TrucoEnv, Player

###
# Setting the environment
###
components = {\
    'players':[\
        Player(index='A',atype='t1'),
        Player(index='B',atype='t2'),
        Player(index='C',atype='t2'),
        Player(index='D',atype='t1'),
        ]
    }

env = TrucoEnv(components, display=True)
state = env.reset()

###
# ADLEAP-MAS MAIN ROUTINE
### 
state = env.reset()

done, max_points = False, 12
while (env.points[0] < max_points) and (env.points[0] < max_points) and not done:
    agent = env.get_adhoc_agent()

    # 1. Importing agent method
    method = env.import_method(agent.type)

    # 2. Reasoning about next action and target
    agent.next_action, _ = method(state, agent)

    # 3. Taking a step in the environment
    next_state, reward, done, _ = env.step(action=agent.next_action)
    state = next_state
    time.sleep(1)

env.close()
###
# THE END - That's all folks :)
###S