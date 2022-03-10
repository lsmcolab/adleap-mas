import gym
import random
import sys
import os
sys.path.append(os.getcwd())

from src.envs.RockSampleEnv import RockSampleEnv, Rock, Agent

rock_1 = Rock(0,(2,2),"good")
rock_2 = Rock(1,(5,2),"good")
rock_3 = Rock(2,(3,7),"bad")
rock_4 = Rock(3,(6,8),"bad")

agent = Agent(0,(3,3),"pomcp")
components = {"rocks":[rock_1,rock_2,rock_3,rock_4],"agents":[agent]}

display = True
env = RockSampleEnv(components=components,dim=10,display=display)
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