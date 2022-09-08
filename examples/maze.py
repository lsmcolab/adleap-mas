###
# Imports
###
import sys
import os
sys.path.append(os.getcwd())

from src.envs.MazeEnv import MazeEnv, Agent

###
# Setting the environment
###
display = True
dim = (10,10)
agent_position = (int(dim[0]/2),int(dim[1]/2))

components = {"agents":[Agent(index= 0, type= 'pomcp')],\
    "black":[(5,5),(3,3),(2,2),(7,7),(9,8),(8,9)]}

env = MazeEnv(agent_position,dim,components,display=display)

###
# ADLEAP-MAS MAIN ROUTINE
###
state = env.reset()
agent = env.get_adhoc_agent()

done, max_episode = False, 200
while env.episode < max_episode and not done:
    # 1. Importing agent method
    agent = env.get_adhoc_agent()
    method = env.import_method(agent.type)

    # 2. Reasoning about next action and target
    agent.next_action, _ = method(state, agent)

    # 3. Taking a step in the environment
    state,_,done,_ = env.step(action=agent.next_action)

env.close()
###
# THE END - That's all folks :)
###
