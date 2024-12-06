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
tag=head_init  # experiment name

num_epochs=100
lr=0.1
lamb=10000

# for wu_wd in 0.0 0.03 0.1 0.3 0.5 1.0; do
#   for seed in 0 1 2; do
#     ./experiments/ewc2.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${wu_nepochs} ${wu_lr} ${wu_wd} ${lr} ${lamb} ${head_init} ${stop_at_task} &
#   done
#   wait
# done
# wait

# without warm-up:
for seed in 0 1 2; do
  ./head_init_exps/ewc2.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} 0 0 0.0 ${lr} ${lamb} zeros ${stop_at_task} linear &
done
wait
