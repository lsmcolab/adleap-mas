from src.reasoning.a_star import a_star_planning
import numpy as np

#####
# LEADER 1 ALGORITHM FROM ALBRECHT AND STONE
#####
# returns the action to lead to furthest task
def l1_planning(env, agent):
	# 1. Choosing a target
	if agent.target is None or env.state[agent.target[0],agent.target[1]] == -1:
		# - choosing a target
		target_position = l1_choose_target_po(env.state, agent)
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

# returns the furthest visible task
# NOTE: For Aldbrecht and Stone, the agent gets the
# furthest visible task with level below of its own.
# In this implementation, as we assume the partial 
# observability, we don't know the task's level.
def l1_choose_target_po(state, agent):
	# 0. Initialising the support variables
	furthest_task_idx, max_distance = -1, -1

	# 1. Searching for max distance item
	visible_tasks = [(x,y) for x in range(state.shape[0]) 
		for y in range(state.shape[1]) if state[x,y] == np.inf]
	for i in range(0, len(visible_tasks)):
		dist = distance(visible_tasks[i],agent.position)
		if dist > max_distance:
			max_distance = dist
			furthest_task_idx = i

	# 2. Verifying the found task
	# a. no task found
	if furthest_task_idx == -1:
		return None
	# b. task found
	else:
		return visible_tasks[furthest_task_idx]

def distance(obj, viewer):
	return np.sqrt((obj[0] - viewer[0])**2 + (obj[1] - viewer[1])**2)