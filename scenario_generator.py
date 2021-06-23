import numpy as np
import random

def get_initial_positions(env_name, dim, nagents, ntasks):
	# verifying if it is possible to create the scenario
	if env_name == "LevelForagingEnv" and (ntasks*9) > (dim**2):
		raise StopIteration

	# getting the random positions
	pos = []
	while len(pos) < (nagents + ntasks):
		x = random.randint(0,dim-1)
		y = random.randint(0,dim-1)
		
		if env_name == "LevelForagingEnv":
			if len(pos) >= nagents:
				if x > 0 and x < dim and y > 0 and y < dim and\
				(x,y) not in pos and (x+1,y) not in pos and\
				(x+1,y+1) not in pos and (x,y+1) not in pos and\
				(x-1,y+1) not in pos and (x-1,y) not in pos and\
				(x-1,y-1) not in pos and (x,y-1) not in pos and\
				(x+1,y-1) not in pos:
					pos.append((x,y))
			else:
				if (x,y) not in pos:
					pos.append((x,y))
		elif env_name == "CaptureEnv":
			if (x,y) not in pos:
				pos.append((x,y))
		else:
			raise NotImplemented

	return pos

def get_env_types(env_name):
	if env_name == "LevelForagingEnv":
		return ['l1','l2','l3']
	elif env_name == "CaptureEnv":
		return ['c1','c2','c3']
	else:
		raise NotImplemented

def get_env_nparameters(env_name):
	if env_name == "LevelForagingEnv":
		return 3
	elif env_name == "CaptureEnv":
		return 2
	else:
		raise NotImplemented
	

def create_LevelForagingEnv(dim, num_agents, num_tasks, partial_observable=False, display=False):
    # 1. Importing the environment and its necessary components
	from src.envs.LevelForagingEnv import LevelForagingEnv, Agent, Task

	# 2. Defining the types and directions of the environment
	types = get_env_types("LevelForagingEnv")
	direction = [0,np.pi/2,np.pi,3*np.pi/2]

	# 3. Getting the initial positions
	random_pos = get_initial_positions("LevelForagingEnv",dim, num_agents, num_tasks)

	# 4. Creating the components
	agents, tasks = [], []

	# a. main agent
	if not partial_observable:
		agents.append(
			Agent(index=str(0), atype='mcts',
			 position=(random_pos[0][0],random_pos[0][1]),
			 direction=random.sample(direction, 1)[0], radius=1.0, angle=1.0,
			 level=random.uniform(0.5, 1)))
	else:
		agents.append(
			Agent(index=str(0), atype='pomcp',
			 position=(random_pos[0][0],random_pos[0][1]),
			 direction=random.sample(direction, 1)[0], radius=random.uniform(0.5,1), angle=random.uniform(0.5,1),
			 level=random.uniform(0.5, 1)))

	# b. teammates and tasks
	for i in range(1, num_agents + num_tasks):
		if (i < num_agents):
			agents.append(\
				Agent(index=str(i), atype=random.sample(types,1)[0], position=(random_pos[i][0],random_pos[i][1]),
					direction=random.sample(direction,1)[0],radius=random.uniform(0.1,1), angle=random.uniform(0.1,1), level=random.uniform(0,1)))
		else:
			tasks.append(Task(str(i), position=(random_pos[i][0],random_pos[i][1]), level=random.uniform(0,1)))

	# c. adding to components dict
	components = {
		'agents': agents,
		'adhoc_agent_index': '0',
		'tasks': tasks}

	# 5. Initialising the environment and returning it
	if partial_observable:
		env = LevelForagingEnv((dim, dim), components,visibility='partial',display=display)
	else:
		env = LevelForagingEnv((dim, dim), components,visibility='full',display=display)
	return env

def create_CaptureEnv(dim, num_agents, num_tasks, partial_observable=False, display=False):
    # 1. Importing the environment and its necessary components
	from src.envs.CaptureEnv import CaptureEnv, Agent, Task

	# 2. Defining the types and directions of the environment
	types = get_env_types('CaptureEnv')
	direction = [0,np.pi/2,np.pi,3*np.pi/2]

	# 3. Getting the initial positions
	random_pos = get_initial_positions('CaptureEnv',dim, num_agents, num_tasks)

	# 4. Creating the components
	agents, tasks = [], []

	# a. main agent
	if not partial_observable:
		agents.append(
			Agent(index=str(0), atype='mcts',
			 position=(random_pos[0][0],random_pos[0][1]),
			 direction=random.sample(direction, 1)[0], radius=1.0, angle=1.0,
			 level=random.uniform(0.5, 1)))
	else:
		agents.append(
			Agent(index=str(0), atype='pomcp',
			 position=(random_pos[0][0],random_pos[0][1]),
			 direction=random.sample(direction, 1)[0], radius=random.uniform(0.5,1), angle=random.uniform(0.5,1),
			 level=None))

	# b. teammates and tasks
	for i in range(1, num_agents + num_tasks):
		if (i < num_agents):
			agents.append(\
				Agent(index=str(i), atype=random.sample(types,1)[0], position=(random_pos[i][0],random_pos[i][1]),
					direction=random.sample(direction,1)[0],radius=random.uniform(0.1,1), angle=random.uniform(0.1,1), level=None))
		else:
			tasks.append(Task(str(i), position=(random_pos[i][0],random_pos[i][1]), level=None))

	# c. adding to components dict
	components = {
		'agents': agents,
		'adhoc_agent_index': '0',
		'tasks': tasks}

	# 5. Initialising the environment and returning it
	if partial_observable:
		env = CaptureEnv((dim, dim), components,visibility='partial',display=display)
	else:
		env = CaptureEnv((dim, dim), components,visibility='full',display=display)
	return env