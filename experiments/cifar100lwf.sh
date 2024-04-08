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
tag=head_after_wu  # experiment name

lamb=1  # best = 10
lamb_mc=0.5
beta=10
gamma=1e-3

num_epochs=100
lr=0.1
wu_lr=0.8
head_init=zeros

for wu_nepochs in 0 20; do
  for seed in 0 1 2; do
    ./experiments/lwf2.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb} ${wu_nepochs} ${wu_lr} ${lr} ${head_init} ${stop_at_task} &
  done
  wait

done
