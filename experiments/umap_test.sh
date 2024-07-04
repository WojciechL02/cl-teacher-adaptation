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
tag=msp_test_umap  # experiment name

lamb=5
num_epochs=100
lr=0.1
wu_lr=0.1
head_init=zeros

# without warm-up:
seed=0
lamb_lwf=1
./experiments/msp.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb} ${lr} ${head_init} ${stop_at_task} &
./experiments/lwf2.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb_lwf} 0 0 0.0 ${lr} ${head_init} ${stop_at_task}