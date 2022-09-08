"""
File to lunch examples/*.py experiments on a HEC cluster
"""
# Importing the necessary packages
from os.path import isdir
import subprocess
import time

# Defining supportive methods
def python_cmd(env,method,exp_type,id=0):
    if env == 'maze' or env == 'rocksample':
        if not isdir("results/"+env+"."+str(id)):
            subprocess.run(["mkdir","results/"+env+"."+str(id)])

        return "python examples/"+env+"_"+exp_type+".py"+\
                                " --exp_num "+str(i)+\
                                " --atype "+method+\
                                " --id "+str(id)
    elif env == 'smartfirebrigade':
        return "python examples/"+env+"_"+exp_type+".py"+\
                                " --exp_num "+str(i)+\
                                " --atype "+method+\
                                " --mode "+str(mode)
    else:
        return "python examples/"+env+"_"+exp_type+".py"+\
                                " --exp_num "+str(i)+\
                                " --atype "+method

# Checking AdLeap-MAS folder integrity
if not isdir("results"):
    subprocess.run(["mkdir","results"])
if not isdir("qsuboutput"):
    subprocess.run(["mkdir","qsuboutput"])
if not isdir("qsuberror"):
    subprocess.run(["mkdir","qsuberror"])
if not isdir("tmp"):
    subprocess.run(["mkdir","tmp"])

# Setting experiments configuration
exp_type = 'smalltest'
env = "tiger"
methods = ['pomcp','ibpomcp','rhopomcp']
nexperiments = 20
scenario_id = [0,1,2,3]
mode = 'control'

# Lunching experiments
for i in range(0,nexperiments):
    for method in methods:
        # environments with different scenarios configuration
        if env == 'maze' or env == 'rocksample':
            for id in scenario_id:
                with open('run.sh','w') as runfile:
                    runfile.write("#$ -S /bin/bash\n\n")
                    runfile.write("#$ -N "+method+"_"+env+str(id)+"_"+str(i)+"\n")
                    runfile.write("#$ -l h_vmem=3G\n")
                    runfile.write("#$ -l h_rt=01:30:00\n")
                    runfile.write("#$ -l node_type=10Geth64G\n\n")
                    runfile.write("source /etc/profile\n")
                    runfile.write("module add anaconda3/wmlce\n")
                    runfile.write("source activate wmlce_env\n\n")
                    runfile.write(python_cmd(env,method,exp_type,id))
        
                subprocess.run(["qsub","-o","qsuboutput/","-e","qsuberror/","run.sh"])
                time.sleep(0.1)
        else:
            # environments with a single scenario configuration
            with open('run.sh','w') as runfile:
                runfile.write("#$ -S /bin/bash\n\n")
                runfile.write("#$ -N "+method+"_"+env+"_"+str(i)+"\n")
                runfile.write("#$ -l h_vmem=3G\n")
                runfile.write("#$ -l h_rt=01:30:00\n")
                runfile.write("#$ -l node_type=10Geth64G\n\n")
                runfile.write("source /etc/profile\n")
                runfile.write("module add anaconda3/wmlce\n")
                runfile.write("source activate wmlce_env\n\n")
                runfile.write(python_cmd(env,method,exp_type))
        
            subprocess.run(["qsub","-o","qsuboutput/","-e","qsuberror/","run.sh"])
            time.sleep(0.1)