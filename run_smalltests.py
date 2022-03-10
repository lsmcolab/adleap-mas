from os.path import isdir
import subprocess
import time

EXP_TYPE = 'smalltest'

env = "maze"
methods = ['pomcp','rhopomcp','ibl']
nexperiments = 200
scenario_id = 0

if not isdir("results"):
    subprocess.run(["mkdir","results"])
if not isdir("qsuboutput"):
    subprocess.run(["mkdir","qsuboutput"])
if not isdir("qsuberror"):
    subprocess.run(["mkdir","qsuberror"])

def python_cmd(env):
    if env == 'maze':
        if not isdir("results/maze."+str(scenario_id)):
            subprocess.run(["mkdir","results/maze."+str(scenario_id)])

        return "python examples/"+env+"_"+EXP_TYPE+".py"+\
                                " --exp_num "+str(i)+\
                                " --atype "+label+\
                                " --id "+str(scenario_id)
    else:
        return "python examples/"+env+"_"+EXP_TYPE+".py"+\
                                " --exp_num "+str(i)+\
                                " --atype "+label

for i in range(0,nexperiments):
    for label in methods:
        with open('run.sh','w') as runfile:
            runfile.write("#$ -S /bin/bash\n\n")
            runfile.write("#$ -N "+label+"_"+str(i)+"\n")
            runfile.write("#$ -l h_vmem=2G\n")
            runfile.write("#$ -l h_rt=04:00:00\n\n")
            runfile.write("source /etc/profile\n")
            runfile.write("module add anaconda3/wmlce\n")
            runfile.write("source activate wmlce_env\n\n")
            runfile.write(python_cmd(env))
        
        subprocess.run(["qsub","-o","qsuboutput/","-e","qsuberror/","run.sh"])
        time.sleep(0.1)