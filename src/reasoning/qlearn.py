import math
import numpy as np 
import random

def create_qtable(actions):
	qtable = {}
	for a in actions:
		qtable[str(a)] = {'qvalue':0.0,'sumvalue':0.0,'trials':0}
	return qtable

def uct_select_action(node,gamma=0.5):
	# 1. Initialising the support values
	maxUCB, maxA = -1, None

	# 2. Checking the best action via UCT algorithm
	for a in node.actions:
		qvalue = node.qtable[str(a)]['qvalue']
		trials = node.qtable[str(a)]['trials']
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

def ibl_select_action(node,gamma):
	# 1. Initialising the support values
	max_quality, maxA = -1, None
	node_entropy = node.entropy
	max_entropy = node.max_entropy

	# 2. Checking the best action via UCT algorithm
	for child in node.children:
		a = child.action

		qvalue = node.qtable[str(a)]['qvalue']
		trials = node.qtable[str(a)]['trials']
		if trials > 0:
			current_ucb = qvalue + gamma *\
			  np.sqrt(np.log(float(node.visits)) / float(trials))

			infoset = {}
			for obs in child.children:
				infoset[obs.state] = obs.visits+1
			action_entropy = entropy(infoset)
			if action_entropy > max_entropy:
				max_entropy = action_entropy
			current_entropy = (1 - gamma) * (action_entropy - node_entropy)/max_entropy

			if (current_ucb + current_entropy) > max_quality:
				max_quality = current_ucb + current_entropy
				maxA = a
		else:
			return a

	# 3. Checking if the best action was found
	if maxA is None:
		maxA = random.sample(node.actions,1)[0]

	# 4. Returning the best action
	return maxA

def entropy(set):
	H = 0
	for x in set:
		Px = set[x]/sum([set[y] for y in set])
		H += Px*math.log(Px)
	return -H