#$ -S /bin/bash

#$ -l h_vmem=4G
#$ -l h_rt=00:10:00

source /etc/profile
module add anaconda3/wmlce
source activate wmlce_env

python experiment.py --env CaptureEnv --estimation AGA --num_agents 7 --num_tasks 20 --dim 20 --num_exp 1 --num_episodes 200 --po False