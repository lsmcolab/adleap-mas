###
# IMPORTS
###
import multiprocessing
import sys
import os

sys.path.append(os.getcwd())
from src.envs.SmartFireBrigadeEnv import SmartFireBrigadeEnv, Agent, Fire

from datetime import datetime
import numpy as np
from multiprocessing import Process, Queue, freeze_support

from src.log import LogFile

###
# THREADING FUNCTIONS
###
def threading_f(agent, env, queue):
    copied_env = env.copy()
    method = copied_env.import_method(agent.type)
    copied_env.components['adhoc_agent_index'] = agent.index
    observable_env = copied_env.get_observation()

    if agent.type == 'pomcp':
        new_action, _ = method(observable_env, agent, max_depth=20, max_it=100)
    else:
        new_action, _ = method(observable_env, agent)
    
    queue.put(new_action)

###
# SMART FIRE BRIGADE ENVIRONMENT SETTINGS
###
TIME_TO_REASON = 1      # 1 seconds
SPAWN_DELAY = 20        # 20 seconds
TIMEOUT = 1*60          # 2 minutes

# Main routine
def main():
    global TIME_TO_REASON, SPAWN_DELAY, TIMEOUT

    # Creating the environment and defining its components
    components = {
        'agents':[\
           Agent(index='A',atype='extinguisher',position=(50,50),direction=((1.5*np.pi)/2),radius=30,angle=np.pi),\
            Agent(index='B',atype='extinguisher',position=(25,25),direction=((1*np.pi)/1),radius=30,angle=np.pi/2),\
            Agent(index='C',atype='extinguisher',position=(25,75),direction=((2*np.pi)/1),radius=30,angle=np.pi/2),\
            Agent(index='D',atype='extinguisher',position=(75,75),direction=((3*np.pi)/2),radius=30,angle=np.pi/2),\
            Agent(index='E',atype='extinguisher',position=(75,25),direction=((1*np.pi)/2),radius=30,angle=np.pi/2),\
                ],
        'adhoc_agent_index':'A',
        'fire':[\
            Fire(position=(50,25), level=3, time_constraint=True),\
            Fire(position=(50,60), level=3, time_constraint=True),\
                ]
        }

    env = SmartFireBrigadeEnv(components=components,dim=(100,100),spawn_delay=SPAWN_DELAY,display=True)
    state = env.reset()
    adhoc_agent = env.get_adhoc_agent()
    actions = {}
    for agent in env.components['agents']:
        actions[agent.index] = None
        
    # Starting Log
    header = ['Time Step (s)','#Fire','Spreading Level','#Agents','Battery Level','Water Level', 'Exploration Level']
    logfile = LogFile('SmartFireBrigadeEnv','test.csv',header)

    ###
    # ADLEAP-MAS MAIN ROUTINE
    ###
    # Starting the experiment routine
    done = False
    all_threads = []
    last_release = datetime.now()
    # Initialising agents threads
    threads, queues = {}, {}
    for agent in env.components['agents']:
        queues[agent.index] = Queue()
        threads[agent.index] = None
    
    count = 0 
    while not done and (datetime.now() - env.start_time).total_seconds() < TIMEOUT:
        # Rendering the environment
        env.render()
            
        # Creating and releasing the agents threads
        if (datetime.now() - last_release).total_seconds() > TIME_TO_REASON:
            for agent in env.components['agents']:
                #print(agent.memory)
                if threads[agent.index] is None:
                    thread = Process(target=threading_f, args=(agent, env, queues[agent.index]))
                    threads[agent.index] = thread
                    threads[agent.index].start()
                    all_threads.append(thread)
                elif not threads[agent.index].is_alive():
                    thread = Process(target=threading_f, args=(agent, env, queues[agent.index]))
                    threads[agent.index] = thread
                    threads[agent.index].start()
                    all_threads.append(thread)

                # Collecting the current results
                if not queues[agent.index].empty():
                    result = queues[agent.index].get()
                    actions[agent.index] = result
                

            last_release = datetime.now()
            data = env.get_logdata()
            logfile.write(None, data)
            
        # Step on environment
        print(actions)
        state, reward, done, info = env.step(actions)
        
    done = True

    # finishing active threads
    for t in all_threads:
        if t.is_alive():
            t.terminate()

    env.close()

# WINDOWS SAFE PARALLEL EXECUTION
if __name__ == '__main__':
    freeze_support()
    main()
###
# THE END - That's all folks :)
###
