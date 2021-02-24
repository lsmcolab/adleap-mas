
def unif_planning(env, agent):
	# 1. Choosing a target

	#target_position = agent.target

	# - planning the action/route to the target
	# if it exists
	next_action = env.action_space.sample()

	return next_action,None