import random


class FundamentalValues:
    def __init__(self,radius_max, radius_min, angle_max, angle_min, level_max, level_min, agent_types, env_dim,
                 estimation_mode):
        self.radius_max = radius_max
        self.radius_min = radius_min
        self.angle_max = angle_max
        self.angle_min = angle_min
        self.level_max = level_max
        self.level_min = level_min
        self.agent_types = agent_types
        self.env_dim = env_dim
        self.estimation_mode = estimation_mode

    def generate_random_parameter(self):
        random_param = Parameter()
        random_param.radius = float(random.uniform(self.radius_min, self.radius_max))  # 'radius'
        random_param.angle = float(random.uniform(self.angle_min, self.angle_max))  # 'angle'
        random_param.level = float(random.uniform(self.level_min, self.level_max))  # 'level'
        return random_param


class Parameter:
    def __init__(self,level=None, angle=None, radius=None):
        self.level = level
        self.angle = angle
        self.radius = radius

    def set_parameters(self, level, angle, radius):
        self.level = float(level)
        self.angle = float(angle)
        self.radius = float(radius)

    def show(self):
        print(self.level,self.angle,self.radius)

########################################################################################################################

