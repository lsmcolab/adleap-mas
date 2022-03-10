import sys
import os
sys.path.append(os.getcwd())

from src.envs.MazeEnv import MazeEnv, Agent

agent = Agent(0,(2,5),"pomcp")
components = {"agents":[agent],"black":[(5,5),(3,3),(2,2),(7,7),(9,8),(8,9)]}

display = True
env = MazeEnv(components=components,dim=10,display=display)
state = env.reset()

done = False
agent = env.get_adhoc_agent()
###
# ADLEAP-MAS MAIN ROUTINE
###
while env.episode < 20 and not done:
    if display:
       env.render()

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
