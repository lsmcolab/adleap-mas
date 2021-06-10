import subprocess
import sys
import os
import time   
###
# FUNCTIONS       
###
# python cmd format functions               
def get_python_cmd(env, estimation, num_agents, num_tasks, dim, num_exp, num_episodes,po):
    return 'python experiment.py'+\
     ' --env ' + str(env) + ' --estimation ' + str(estimation) + \
     ' --num_agents ' + str(num_agents) + ' --num_tasks ' + str(num_tasks) + \
     ' --dim ' + str(dim) + ' --num_exp ' + str(num_exp) + ' --num_episodes ' + str(num_episodes) + ' --po ' + str(po)

python_cmd = get_python_cmd(env = 'CaptureEnv', estimation = 'AGA', num_agents = 5,
                             num_tasks = 10, dim = 20, num_exp = 1, num_episodes = 200,po=False)

###
# SCRIPT
###
# writing the bash file                     
with open("run.sh", "w") as bashfile:           
    bashfile.write("#$ -S /bin/bash\n\n")       
    bashfile.write("#$ -l h_vmem=4G\n")         
    bashfile.write("#$ -l h_rt=00:10:00\n\n")  
    bashfile.write("source /etc/profile\n")     
    bashfile.write("module add anaconda3/wmlce\n")                              
    bashfile.write("source activate wmlce_env\n\n")                             
    bashfile.write(python_cmd)