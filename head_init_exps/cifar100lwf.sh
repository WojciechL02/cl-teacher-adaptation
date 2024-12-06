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

lamb=1  # best = 10
lamb_mc=0.5
beta=10
gamma=1e-3

num_epochs=100
lr=0.1


# for wu_wd in 0.0 0.03 0.3; do
#   for seed in 0 1 2; do
#     ./experiments/lwf2.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb} ${wu_nepochs} ${wu_lr} ${wu_wd} ${lr} ${head_init} ${stop_at_task} &
#   done
#   wait
# done
# wait

#without warm-up:
for seed in 0 1 2; do
  ./head_init_exps/lwf2.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb} 0 0 0.0 ${lr} zeros ${stop_at_task} linear &
done
wait

# for seed in 0 1 2; do
#   ./experiments/lwf2.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lamb} 0 0 0.0 ${lr} ${head_init} ${stop_at_task} linear &
# done
# wait
