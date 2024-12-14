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
dataset=imagenet_subset_kaggle
network=resnet18
# tag=lie_bracket
tag=scr_thesis  # experiment name

num_epochs=100
lr=0.1
head_init=zeros
exemplars=2000
temperature=0.1
batch_size=256

# for seed in 0 1 2; do
./experiments/scr.sh 0 2 ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lr} ${head_init} ${stop_at_task} ${exemplars} ${temperature} ${batch_size}
# done
# wait


# for seed in 0 1 2; do
# ./experiments/supcon.sh 0 0 ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lr} ${head_init} ${stop_at_task} 0 50 ${temperature} 256
# done
# wait

# ./experiments/ft_nmc.sh 0 0 ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} 100 0.1 ${head_init} ${stop_at_task} 1 20
# ./experiments/ft_nmc.sh 0 0 ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} 100 0.1 ${head_init} ${stop_at_task} 1 50