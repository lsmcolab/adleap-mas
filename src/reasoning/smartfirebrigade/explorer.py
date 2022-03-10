import numpy as np

def angle_of_gradient(obj, viewer, direction):
    xt = (obj[0] - viewer[0])
    yt = (obj[1] - viewer[1])

    x = np.cos(direction) * xt + np.sin(direction) * yt
    y = -np.sin(direction) * xt + np.cos(direction) * yt
    if (y == 0 and x == 0):
        return 0
    return np.arctan2(y, x)

def explorer_planning(env,agent):
    next_action = None
    pos, dir = agent.position, agent.direction
    exploration_velocity = 4
    communicate_chance = 0.5
    if len(env.state['agents']) > 0 and np.random.random()<communicate_chance:
        next_action = env.get_action('COMMUNICATE')
        return next_action,None

    if pos[0] < (env.dim[0]/2) and pos[1] < (env.dim[1]/2):
        if agent.velocity > exploration_velocity:
            next_action = env.get_action('DOWN')
            return next_action, None
        if not agent.direction < 0.1*np.pi and not agent.direction > 3.9*np.pi/2:
            next_action = env.get_action('LEFT')
            return next_action, None

    elif pos[0] > (env.dim[0]/2) and pos[1] < (env.dim[1]/2):
        if agent.velocity > exploration_velocity:
            next_action = env.get_action('DOWN')
            return next_action, None
        if not 0.9*np.pi/2 < agent.direction < 1.1*np.pi/2:
            next_action = env.get_action('LEFT')
            return next_action, None

    elif pos[0] > (env.dim[0]/2) and pos[1] > (env.dim[1]/2):
        if agent.velocity > exploration_velocity:
            next_action = env.get_action('DOWN')
            return next_action, None
        if not 0.9*np.pi < agent.direction < 1.1*np.pi:
            next_action = env.get_action('LEFT')
            return next_action, None
            
    elif not 2.9*np.pi/2 < agent.direction < 3.1*np.pi/2:
        if agent.velocity > exploration_velocity:
            next_action = env.get_action('DOWN')
            return next_action, None
        else:
            next_action = env.get_action('LEFT')
            return next_action, None

    if agent.velocity < exploration_velocity:
        next_action = env.get_action('UP')
        return next_action, None
    else:
        next_action = env.get_action('DOWN')
        return next_action, None