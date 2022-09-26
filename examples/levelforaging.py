###
# IMPORTS
###
import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from src.envs.LevelForagingEnv import LevelForagingEnv,Agent,Task

###
# Setting the environment
###
display = True
dim = (10,10)
visibility = 'partial'
estimation_method = None

components = {
    'agents' : [
            Agent(index='A',atype='pomcp',position=(1,1),direction=1*np.pi/2,radius=0.25,angle=1,level=1.0),
            Agent(index='1',atype='l1',position=(8,1),direction=1*np.pi/2,radius=0.25,angle=1,level=0.2),
            Agent(index='2',atype='l2',position=(1,8),direction=1*np.pi/2,radius=0.25,angle=1,level=0.4),
            Agent(index='3',atype='l3',position=(8,8),direction=1*np.pi/2,radius=0.25,angle=1,level=0.6)
                ],
    'adhoc_agent_index' : 'A',
    'tasks' : [
            Task(index='0',position=(8,8),level=1.0),
            Task(index='1',position=(5,5),level=0.9),
            Task(index='2',position=(0,0),level=0.7),
            Task(index='3',position=(9,1),level=1.0)
                ]
}

env = LevelForagingEnv(shape=dim,components=components,visibility=visibility,display=display)

###
# ADLEAP-MAS MAIN ROUTINE
###
adhoc_agent = env.get_adhoc_agent()
state = env.reset()

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
