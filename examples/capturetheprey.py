###
# IMPORTS
###
import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from src.envs.CaptureEnv import CaptureEnv,Hunter,Prey

###
# Setting the environment
###
display = True
dim = (10,10)
estimation_method = None

components = {
    'hunters' : [
            Hunter(index='A',atype='pomcp',position=(1,1),direction=1*np.pi/2,radius=1,angle=1),
            Hunter(index='1',atype='c1',position=(3,3),direction=1*np.pi/2,radius=1,angle=1),
            Hunter(index='2',atype='c2',position=(4,5),direction=1*np.pi/2,radius=1,angle=1),
            Hunter(index='3',atype='c3',position=(7,7),direction=1*np.pi/2,radius=1,angle=1)
                ],
    'adhoc_agent_index' : 'A',
    'preys' : [
            Prey(index='0',position=(8,8)),
            Prey(index='1',position=(5,5)),
            Prey(index='2',position=(0,0)),
            Prey(index='3',position=(9,1))
                ]
}

env = CaptureEnv(shape=dim,components=components,display=display)

###
# ADLEAP-MAS MAIN ROUTINE
###
adhoc_agent = env.get_adhoc_agent()
state = env.reset()

# TODO: Fix the black screen at the first rendering

done, max_episode = False, 200
while env.episode < max_episode and not done:
    # 1. Importing agent method
    method = env.import_method(adhoc_agent.type)

    # 2. Reasoning about next action and target
    adhoc_agent.next_action, _ = method(state, adhoc_agent)

    # 3. Taking a step in the environment
    state,_,done,_ = env.step(action=adhoc_agent.next_action)

env.close()
###
# THE END - That's all folks :)
###
