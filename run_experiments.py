import subprocess
import os
from re import search
import time   

from scenario_generator import create_LevelForagingEnv, create_CaptureEnv

###
# FUNCTIONS       
###
# check the files and folders integrity
def check_expenv():
    # chekcing result and log folders
    if not os.path.isdir("./results"):
        os.mkdir("./results")
    if not os.path.isdir("./qsuberror"):
        os.mkdir("./qsuberror")
    if not os.path.isdir("./qsuboutput"):
        os.mkdir("./qsuboutput")
    if not os.path.isdir("./bashlog"):
        os.mkdir("./bashlog")

    # checking map folders
    if not os.path.isdir("./src/envs/maps"):
        os.mkdir("./src/envs/maps")
    if not os.path.isdir("./src/envs/maps/CaptureEnv"):
        os.mkdir("./src/envs/maps/CaptureEnv")
    if not os.path.isdir("./src/envs/maps/LevelForagingEnv"):
        os.mkdir("./src/envs/maps/LevelForagingEnv")

# load the configurations from a csv file
def get_configurations():
    configurations = []

    with open('experiment_configuration.csv','r') as config_file:
        for line in config_file:
            if search('^#',line) is None and search('^$',line[0]) is None:
                print(line)
                config = line.split(',')

                configurations.append({'mode':None, 'num_exp':None, 'num_episodes':None,\
                 'env':None, 'num_agents':None, 'num_tasks':None, 'dim':None})

                configurations[-1]['mode'] = config[0]
                configurations[-1]['num_exp'] = config[1]
                configurations[-1]['num_episodes'] = config[2]
                configurations[-1]['env'] = config[3]
                configurations[-1]['num_agents'] = config[4]
                configurations[-1]['num_tasks'] = config[5]
                configurations[-1]['dim'] = config[6]
        
    return configurations

# python cmd format functions               
def get_python_cmd(mode, env, estimation, num_agents, num_tasks, dim, num_exp, num_episodes):
    return 'python experiment_'+ str(mode) +'.py'+\
     ' --env ' + str(env) + ' --estimation ' + str(estimation) + \
     ' --num_agents ' + str(num_agents) + ' --num_tasks ' + str(num_tasks) + \
     ' --dim ' + str(dim) + ' --num_exp ' + str(num_exp) + ' --num_episodes ' + str(num_episodes) + \
     ' --po False --display False' 

# create qsubfile for submission
def create_qsubfile(mode,estimation,num_agents,num_tasks,dim,experiment_id):
    fname = mode+estimation+"_a"+str(num_agents)+"_i"+str(num_tasks)+"_d"+str(dim)+"_exp"+str(experiment_id)
    with open("run.sh", "w") as bashfile:           
        bashfile.write("#$ -S /bin/bash\n\n")   
        bashfile.write("#$ -N "+fname+"\n")   
        bashfile.write("#$ -l h_vmem=6G\n")         
        bashfile.write("#$ -l h_rt=02:00:00\n\n")  
        bashfile.write("source /etc/profile\n")     
        bashfile.write("module add anaconda3/wmlce\n")                              
        bashfile.write("source activate wmlce_env\n\n")                             
        bashfile.write(python_cmd)

    time.sleep(0.25)
###
# SCRIPT
###
# 1. Check environment integrity
check_expenv()

# 2. Loading experiment settings
configurations = get_configurations()

for config in configurations:
    # 3. Setting the configurations
    mode = config['mode']
    num_exp = int(config['num_exp'])
    num_episodes = int(config['num_episodes'])
    env = config['env']
    num_agents = int(config['num_agents'])

    num_tasks = int(config['num_tasks'])
    dim = int(config['dim'])

    
    # 4. Creating the scenarios
    for experiment_id in range(num_exp):
        if env == 'LevelForagingEnv':
            create_LevelForagingEnv(dim, num_agents, num_tasks, num_exp=experiment_id)
        elif env == 'CaptureEnv':
            create_CaptureEnv(dim, num_agents, num_tasks, num_exp=experiment_id)
        else:
            raise NotImplemented

    # 5. Stating the remote experiment
    for experiment_id in range(num_exp):
        for estimation in ['AGA','ABU',"OEATA"]:
            # defining the experiment parameters
            python_cmd = get_python_cmd(mode,env,estimation,num_agents,
                        num_tasks,dim,experiment_id,num_episodes)

            # writing the bash file
            create_qsubfile(mode,estimation,num_agents,num_tasks,dim,experiment_id)

            # submiting the job
            subprocess.run(["qsub","-o","qsuboutput/","-e","qsuberror/","run.sh"])
            time.sleep(0.25)