from a_star import a_star_planning
import numpy as np

#####
# LEADER 5 ALGORITHM
#####
# returns the action to lead to the first task found
def l5_planning(env, agent):
	# 1. Choosing a target
	if agent.target is None or env.state[agent.target[0],agent.target[1]] == -1:
		# - choosing a target
		target_position = l5_choose_target(env.state)
		agent.target = target_position
	else:
		target_position = agent.target

	# - planning the action/route to the target
	# if it exists
	if target_position is not None:
		next_action = a_star_planning(env.state, env.state.shape[0], env.state.shape[1],
									 	env.action_space, agent.position, target_position)
	# else, take a random action
	else:
		next_action = env.action_space.sample()

	# 2. Verifying if the agent can complete a task
	if agent.direction == np.pi/2:
		pos = (agent.position[0],agent.position[1]+1)
	elif agent.direction == 3*np.pi/2:
		pos = (agent.position[0],agent.position[1]-1)
	elif agent.direction == 0:
		pos = (agent.position[0]+1,agent.position[1])
	elif agent.direction == np.pi:
		pos = (agent.position[0]-1,agent.position[1])

	if pos == target_position:
		#target_position = None
		agent.target = target_position
		return 4, target_position

	return next_action,target_position

# returns the first task found
def l5_choose_target(state):
	# 1. Searching tasks
	visible_tasks = [(x,y) for x in range(state.shape[0]) 
		for y in range(state.shape[1]) if state[x,y] == np.inf]

	# 2. Verifying the found task
	# a. no task found
	if len(visible_tasks) == 0:
		return None
	# b. task found
	else:
		return visible_tasks[0]