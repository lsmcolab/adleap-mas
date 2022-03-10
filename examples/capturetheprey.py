###
# IMPORTS
###
import sys
import os
import numpy as np

sys.path.append(os.getcwd())

from src.envs.CaptureEnv import CaptureEnv,Agent,Task
from gym.wrappers import Monitor,RecordVideo
###
# CAPTURE THE PREY ENVIRONMENT SETTINGS
###
components = {
    'agents' : [
            Agent(index='A',atype='c1',position=(1,1),direction=1*np.pi/2,radius=0.5,angle=0.5), 
            Agent(index='1',atype='c1',position=(3,8),direction=0*np.pi/2,radius=0.6,angle=1.0), 
            Agent(index='2',atype='c2',position=(8,3),direction=3*np.pi/2,radius=0.7,angle=0.5), 
            Agent(index='3',atype='c3',position=(5,6),direction=2*np.pi/2,radius=0.5,angle=0.7)
                ],
    'adhoc_agent_index' : 'A',
    'tasks' : [
            Task(index='0',position=(8,8)),
            Task(index='1',position=(5,5)),
            Task(index='2',position=(0,0)),
            Task(index='3',position=(9,1))
                ]
}


dim = (10,10)
display = True
visibility = 'full'
estimation_method = None

env = CaptureEnv(shape=dim,components=components,visibility=visibility,display=display)
state = env.reset()
adhoc_agent = env.get_adhoc_agent()
done = False
###
# ADLEAP-MAS MAIN ROUTINE
###
while not done and env.episode < 30:
    env.render()

    # 1. Importing agent method
    method = env.import_method(adhoc_agent.type)

    # 2. Reasoning about next action and target
    if adhoc_agent.type == 'pomcp' or adhoc_agent.type=='mcts':
        adhoc_agent.next_action, adhoc_agent.target = method(state, adhoc_agent, estimation_algorithm=estimation_method)
    else:
        adhoc_agent.next_action, adhoc_agent.target = method(state, adhoc_agent)

    # 3. Taking a step in the environment
    state,reward,done,info = env.step(adhoc_agent.next_action)

env.close()
###
# THE END - That's all folks :)
###