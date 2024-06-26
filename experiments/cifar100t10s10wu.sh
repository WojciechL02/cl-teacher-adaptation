#!/bin/bash

#SBATCH --time=48:00:00   # walltime
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

lamb=10
lamb_mc=0.5
beta=10
gamma=1e-3

seed=0
num_epochs=100
lr=0.1
wu_nepochs=20

for wu_lr in 1.0 1.2 1.5; do
  ./experiments/lwf1.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb} ${wu_nepochs} ${wu_lr} ${lr} &
done
wait
