import random as rd

def level_foraging_uniform_estimation(env):
    for agent in env.components['agents']:
        if agent != env.get_adhoc_agent() and\
         None in [agent.level,agent.radius,agent.angle,agent.type]:
            agent.level = rd.uniform(0,1)
            agent.radius = rd.uniform(0,1)
            agent.angle = rd.uniform(0,1)
            agent.type = rd.sample(['l1','l2','l3','l4'],1)[0]
        
    for task in env.components['tasks']:
        if task.level is None:
            task.level = rd.uniform(0,1)

    return env

def truco_uniform_estimation(env):
    if env.visibility == 'partial':
        for player in env.components['player']:
            if player != env.components['player'][env.current_player]:
                player.hand = []
                while len(player.hand) < 3:
                    player.hand.append(env.components['cards in game'].pop(0))
                player.type = rd.sample(['t1','t2','t3'],1)[0]
                
    env.visibility = 'full'
    return env