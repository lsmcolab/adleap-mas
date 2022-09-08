import numpy as np

def angle_of_gradient(obj, viewer, direction):
    xt = (obj[0] - viewer[0])
    yt = (obj[1] - viewer[1])

    x = np.cos(direction) * xt + np.sin(direction) * yt
    y = -np.sin(direction) * xt + np.cos(direction) * yt
    if (y == 0 and x == 0):
        return 0
    return np.arctan2(y, x)

def get_closest(fires,pos):
    min_dist, min_dist_index = np.inf, 0
    for i in range(len(fires)):
        fire_dist = np.linalg.norm(np.array(pos) - np.array(fires[i]))
        if fire_dist < min_dist:
            min_dist = fire_dist
            min_dist_index = i
    return fires[min_dist_index],min_dist

def extinguisher_planning(env, agent):
    next_action = None
    pos, dir = agent.position, agent.direction
    state = env.state
    if len(state['fire']) >= 1:
        fire_pos, fire_dist = get_closest(state['fire'], pos)
        if env.action_mode == 'policy':
            return 1, fire_pos
        #checking the distance
        if 10 > fire_dist:
            next_action = env.get_action('EXTINGUISH')
            return next_action, fire_pos
        # go towards the fire
        else:
            if pos[0] > fire_pos[0]:
                return env.get_action('WEST'), fire_pos
            elif pos[0] < fire_pos[0]:
                return env.get_action('EAST') , fire_pos 
            elif pos[1] > fire_pos[1]:
                return env.get_action('SOUTH'), fire_pos
            elif pos[1] < fire_pos[1]:
                return env.get_action('NORTH'), fire_pos
            else:
                agent.target = None, None

    return env.get_action('NOOP'), None