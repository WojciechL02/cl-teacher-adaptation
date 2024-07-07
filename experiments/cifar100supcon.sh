#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=3   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=10
nc_first_task=10
stop_at_task=2  # default = 0
dataset=cifar100_icarl
network=resnet32
tag=nmc3_num_exemp_per_C  # experiment name

num_epochs=100
lr=0.05
head_init=zeros

# without warm-up:
exemplars=50
temperature=0.1

# for seed in 0 1 2; do
#   ./experiments/supcon.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lr} ${head_init} ${stop_at_task} 1 ${exemplars} ${temperature} &
# done
# wait

./experiments/supcon.sh 0 0 ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lr} ${head_init} ${stop_at_task} 1 ${exemplars} ${temperature}


# for seed in 0 1 2; do
#   ./experiments/supcon.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lr} ${head_init} ${stop_at_task} 0 ${exemplars} ${temperature} &
# done
# wait
