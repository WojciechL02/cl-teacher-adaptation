#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=3   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=5
nc_first_task=20
stop_at_task=0  # default = 0
dataset=cifar100_icarl
network=resnet18
tag=figure1  # experiment name

num_epochs=100
lr=0.1
exemplars=2000
head_init=zeros

# without warm-up:

for seed in 0 1 2; do
  ./experiments/ft2.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} 0 0 0.0 ${lr} ${head_init} ${stop_at_task} ${exemplars} &
done
wait
