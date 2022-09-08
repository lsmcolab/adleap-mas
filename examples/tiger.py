###
# Imports
###
import sys
import os
sys.path.append(os.getcwd())

from src.envs.TigerEnv import TigerEnv, Agent

###
# Setting the environment
###
components = {'agents':[Agent(index= 0, type= 'pomcp')]}
tiger_pos = 'left'

env = TigerEnv(components, tiger_pos)

###
# ADLEAP-MAS MAIN ROUTINE
### 
state = env.reset()
agent = env.get_adhoc_agent()

done, max_episode = False, 20
while env.episode < max_episode and not done:
	# 1. Importing agent method
	method = env.import_method(agent.type)

	# 2. Reasoning about next action and target
	agent.next_action, _ = method(state, agent)

	# 3. Taking a step in the environment
	next_state, reward, done, _ = env.step(action=agent.next_action)
	state = next_state
	
	print(env.episode, '| Tiger pos: ',env.tiger_pos,\
					   '| Action: %6s' % state.state['action'],\
					   '| Obs: ',state.state['obs'])
env.close()
###
# THE END - That's all folks :)
###