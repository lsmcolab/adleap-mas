def random_planning(env, agent):
    next_action = env.action_space.sample()
    return next_action, None