#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=3   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

gpu=$1
seed=$2
tag=$3
dataset=$4
num_tasks=$5
nc_first_task=$6
network=$7
num_epochs=$8
lr=${9:-0.1}
head_init=${10}
stop_at_task=${11:-0}

exp_name="cifar100t${num_tasks}s${nc_first_task}_${tag}_hz_lr${lr}"
result_path="results/${tag}/msphl_hz_${seed}"
python3 src/main_incremental.py \
  --exp-name ${exp_name} \
  --gpu ${gpu} \
  --datasets ${dataset} \
  --num-tasks ${num_tasks} \
  --nc-first-task ${nc_first_task} \
  --network ${network} \
  --use-test-as-val \
  --lr ${lr} \
  --nepochs ${num_epochs} \
  --batch-size 128 \
  --seed ${seed} \
  --log disk wandb \
  --results-path ${result_path} \
  --tags ${tag} \
  --scheduler-milestones \
  --approach msphl \
  --stop-at-task ${stop_at_task} \
  --head-init-mode ${head_init} \
  --num-exemplars 2000
