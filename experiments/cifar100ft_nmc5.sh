#!/bin/bash

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=16   # number of processor cores (i.e. tasks)
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
bsz=128
lr=0.1
exemplars=2000
head_init=zeros

for seed in 0 1 2; do
  ./experiments/ft_nmc_bp.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lr} ${head_init} ${stop_at_task} 1 ${exemplars} ${bsz} &
done
wait

# ./experiments/ft_nmc.sh 0 0 ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} 100 0.1 ${head_init} ${stop_at_task} 0 2000 &
# done
# wait

# exemplars=50
# for seed in 0 1 2; do
# ./experiments/ft_nmc.sh 0 0 ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lr} ${head_init} ${stop_at_task} 1 50
# done
# wait
