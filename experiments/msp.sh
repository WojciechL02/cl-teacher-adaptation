#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=3   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

gpu=0
seed=0
tag=msp_test
dataset=cifar100_icarl
num_tasks=10
nc_first_task=10
network=resnet32
num_epochs=20
lamb=5
lr=0.1
head_init=zeros
stop_at_task=2

exp_name="cifar100t${num_tasks}s${nc_first_task}_${tag}_hz"
result_path="results/${tag}/msp_hz_${lamb}_${seed}"
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
  --approach msp \
  --stop-at-task ${stop_at_task} \
  --lamb ${lamb} \
  --head-init-mode ${head_init} \
  --num-exemplars 2000
