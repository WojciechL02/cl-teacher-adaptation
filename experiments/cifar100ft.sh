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
tag=figure1  # experiment name

num_epochs=100
lr=0.1
bsz=128
wu_epochs=50
wu_lr=0.2
wu_wd=0.0
exemplars=2000
head_init=zeros

# without warm-up:
for seed in 0 1 2; do
  ./experiments/ft2.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${wu_epochs} ${wu_lr} ${wu_wd} ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} &
done
wait
