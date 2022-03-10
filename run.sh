#$ -S /bin/bash

#$ -N pomcp_0
#$ -l h_vmem=2G      
#$ -l h_rt=04:00:00 

source /etc/profile
module add anaconda3/wmlce
source activate wmlce_env

python examples/levelforaging_smalltest.py --exp_num 0 --label pomcp