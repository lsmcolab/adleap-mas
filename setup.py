from setuptools import setup

#####
# Major Environment
#####
setup(name='AdhocReasoningEnv-v1',
      version='1.0.0',
      install_requires=['gym']
)

#####
# Toy Problems Environments
#####
setup(name='MazeEnv-v1',
      version='1.0.0',
      install_requires=['gym','numpy']
)

setup(name='RockSampleEnv-v1',
      version='1.0.0',
      install_requires=['gym','numpy']
)

setup(name='TigerEnv-v1',
      version='1.0.0',
      install_requires=['gym','numpy'],
)

#####
# Ad-hoc Teamwork Environment
#####
setup(name='CaptureEnv-v1',
      version='1.0.0',
      install_requires=['gym','numpy']
)

setup(name='LevelForagingEnv-v1',
      version='1.0.0',
      install_requires=['gym','numpy']
)

#####
# Realistic Scenarios
#####
setup(name='SmartFireBrigadeEnv-v1',
      version='1.0.0',
      install_requires=['gym','numpy']
)

#####
# Card Games
#####
setup(name='TrucoEnv-v1',
      version='1.0.0',
      install_requires=['gym']  # And any other dependencies needs
)