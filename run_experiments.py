import subprocess
import sys
import os
import time   
###
# FUNCTIONS       
###
# python cmd format functions               
def get_python_cmd(env, estimation, num_agents, num_tasks, dim, num_exp, num_episodes):
    return 'python experiment.py'+\
     ' --env ' + str(env) + ' --estimation ' + str(estimation) + \
     ' --num_agents ' + str(num_agents) + ' --num_tasks ' + str(num_tasks) + \
     ' --dim ' + str(dim) + ' --num_exp ' + str(num_exp) + ' --num_episodes ' + str(num_episodes) + \
     ' --po False --display False' 

###
# SCRIPT
###
num_exp = 100
num_episodes = 200
for experiment_id in range(num_exp):
    for num_agents in [7]:#5,7,10
        for num_tasks in [20]:#20,25,30
            for dim in [20]:#20,25,30
                for estimation in ['AGA','ABU',"OEATA"]:
                    # defining the experiment parameters
                    python_cmd = get_python_cmd('CaptureEnv',estimation,num_agents,
                                num_tasks,dim,experiment_id,num_episodes)

                    # writing the bash file
                    fname = estimation+"_a"+str(num_agents)+"_i"+str(num_tasks)+"_d"+str(dim)+"_exp"+str(experiment_id)
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

                    # submiting the job
                    subprocess.run(["qsub","-o","qsuboutput/","-e","qsuberror/","run.sh"])
                    time.sleep(0.25)
