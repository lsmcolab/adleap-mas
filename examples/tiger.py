import sys
import random
import os

sys.path.append(os.getcwd())

from src.envs.TigerEnv import TigerEnv, Agent

survived = 0
counter = 0
display = False
discount_factor = 0.75
rewards = []

###
# ADLEAP-MAS MAIN ROUTINE
###
while counter < 1000:
	cum_rew = 0
	print(counter)
	agent = Agent(0,"pomcp")
	components = {"agents":[agent]}
	env = TigerEnv(components=components,tiger_pos='left',display=display)  
	state = env.reset()
	gamma = 1
	done = False  
	while not done and env.episode < 40:
		if display:
			env.render()

		# 1. Importing agent method
		agent = env.get_adhoc_agent()
		method = env.import_method(agent.type)

		# 2. Reasoning about next action and target
		agent.next_action, _ = method(state, agent)
#		agent.next_action = random.sample([0,1,2],1)[0]

		# 3. Taking a step in the environment
		next_state,rew,done,_ = env.step(action=agent.next_action)
		cum_rew += gamma*rew
		gamma = gamma*discount_factor	
		state = next_state
	
	counter+=1
	rewards.append(cum_rew)

print(sum(rewards)/len(rewards))
env.close()
###
# THE END - That's all folks :)
###