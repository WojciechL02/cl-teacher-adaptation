#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=3   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=10
nc_first_task=10
stop_at_task=5  # default = 0
dataset=cifar100_icarl
network=resnet32
tag=nmc_t5  # experiment name

num_epochs=100
lr=0.1
head_init=zeros


#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=3   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=10
nc_first_task=10
stop_at_task=0  # default = 0
dataset=cifar100_icarl
network=resnet32
tag=nmc3_num_exemp_per_C_t10  # experiment name

num_epochs=100
lr=0.1
head_init=zeros

# without warm-up:
exemplars=50
for seed in 0 1 2; do
  ./experiments/ft2.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} 0 0 0.0 ${lr} ${head_init} ${stop_at_task} ${exemplars} &
done
wait
