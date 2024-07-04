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
tag=nmc1  # experiment name

num_epochs=100
lr=0.1
lamb=5
head_init=zeros

# without warm-up:
for seed in 0 1 2; do
  ./experiments/msp_nmc.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lr} ${head_init} ${stop_at_task} ${lamb} 1 &
done
wait

# without warm-up:
for seed in 0 1 2; do
  ./experiments/msp_nmc.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lr} ${head_init} ${stop_at_task} ${lamb} 0 &
done
wait
