#$ -S /bin/bash

#$ -N AGA_a7_i20_d40_exp2
#$ -l h_vmem=8G
#$ -l h_rt=02:00:00

source /etc/profile
module add anaconda3/wmlce
source activate wmlce_env

python experiment_.py --env LevelForagingEnv --estimation OEATA --num_agents 7 --num_tasks 20 --dim 40 --num_exp 2 --num_episodes 200 --po False --display False