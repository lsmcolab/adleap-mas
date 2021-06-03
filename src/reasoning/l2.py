from a_star import a_star_planning
import numpy as np
import random as rd

#####
# LEADER 2 ALGORITHM FROM ALBRECHT AND STONE
#####
# returns the action to lead to task with highest sum of coordinates
def l2_planning(env, agent):
	# 1. Choosing a target
	if agent.target is None or env.state[agent.target[0],agent.target[1]] == -1:
		# - choosing a target
		target_position = l2_choose_target_po(env.state, env.action_space, agent)
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
		target_position = None
		agent.target = target_position
		return 4, target_position

	return next_action,target_position

# returns the furthest visible task
# NOTE: For Aldbrecht and Stone, the agent gets the
# visible task with the highest level below of its own.
# In this implementation, as we assume the partial 
# observability, we don't know the task's level.
# Therefore, the agent choose the task with the
# highest sum of position coordinates.
def l2_choose_target_po(state, action_space, agent):
	# 0. Initialising the support variables
	highest_sum_task_id, max_sum = -1, -1

	# 1. Searching for highest sum task
	visible_tasks = [(x,y) for x in range(state.shape[0]) 
		for y in range(state.shape[1]) if state[x,y] == np.inf]

	for i in range(0, len(visible_tasks)):
		sum_value = sum(visible_tasks[i])
		if sum_value > max_sum:
			max_sum = sum_value
			highest_sum_task_id = i

	# 2. Verifying the found task
	# a. no task found
	if highest_sum_task_id == -1:
		return None
	# b. task found
	else:
		return visible_tasks[highest_sum_task_id]