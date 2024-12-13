#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=16   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=10
nc_first_task=10
stop_at_task=0  # default = 0
dataset=cifar100_icarl
network=resnet18
tag=lie_bracket_3  # experiment name

num_epochs=100
bsz=128
lr=0.1
head_init=zeros
seed=0
exemplars=2000
lamb=0.15
h=0.05


./ft_lb1.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb} ${lr} ${head_init} ${stop_at_task} linear ${h} v3
# ./ft_lb.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb} ${lr} ${head_init} ${stop_at_task} linear 0.1 v7 &
# ./ft_lb.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb} ${lr} ${head_init} ${stop_at_task} linear 0.01 v7 &
# ./experiments/ssil.sh 0 0 ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} 1 0 0 0.0 0.05 ${head_init} ${stop_at_task} linear

# done
# wait

