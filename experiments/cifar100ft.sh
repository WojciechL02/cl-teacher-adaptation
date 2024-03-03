#!/bin/bash

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=3   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=10
nc_first_task=10
dataset=cifar100_icarl
network=resnet32
tag=cifar100t${num_tasks}s${nc_first_task}

seed=0
num_epochs=100
wu_nepochs=0
wu_lr=0.8

for lr in 0.1 0.05, 0.01; do
  ./experiments/ft1.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${wu_nepochs} ${wu_lr} ${lr} &
done
wait

#head_init=zeros
#
#for wu_nepochs in 0 20; do
#  ./experiments/ft2.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${wu_nepochs} ${wu_lr} ${lr} ${head_init} &
#done
#wait
