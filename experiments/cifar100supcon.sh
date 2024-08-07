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
network=resnet18
tag=supcon_distil  # experiment name

num_epochs=200
lr=0.1
head_init=zeros

# without warm-up:
exemplars=2000
temperature=0.07

for seed in 0 1 2; do
  ./experiments/supcon.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lr} ${head_init} ${stop_at_task} 1 ${exemplars} ${temperature} 128 &
done
wait


# for seed in 0 1 2; do
# ./experiments/supcon.sh 0 0 ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lr} ${head_init} ${stop_at_task} 0 50 ${temperature} 256
# done
# wait

# ./experiments/ft_nmc.sh 0 0 ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} 100 0.1 ${head_init} ${stop_at_task} 1 20
# ./experiments/ft_nmc.sh 0 0 ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} 100 0.1 ${head_init} ${stop_at_task} 1 50
