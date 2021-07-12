#$ -S /bin/bash

#$ -l h_vmem=4G
#$ -l h_rt=00:10:00

source /etc/profile
module add anaconda3/wmlce
source activate wmlce_env

python scenario_generator.py --fixed true