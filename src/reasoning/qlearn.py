import numpy as np 
import random

def create_qtable(actions):
	qtable = {}
	for a in actions:
		qtable[a] = {'qvalue':0.0,'sumvalue':0.0,'trials':0}
	return qtable

def uct_select_action(node,gamma=0.5):
	# 1. Initialising the support values
	maxUCB, maxA = -1, None

	# 2. Checking the best action via UCT algorithm
	for a in node.actions:
		qvalue = node.qtable[a]['qvalue']
		trials = node.qtable[a]['trials']
		if trials > 0:
			current_ucb = qvalue + gamma *\
			  np.sqrt(np.log(float(node.visits)) / float(trials))

			if current_ucb > maxUCB:
				maxUCB = current_ucb
				maxA = a
		else:
			return a

	# 3. Checking if the best action was found
	if maxA is None:
		maxA = random.sample(node.actions,1)[0]

	# 4. Returning the best action
	return maxA