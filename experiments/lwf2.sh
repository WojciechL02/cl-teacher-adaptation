#!/bin/bash

set -e

gpu=$1
seed=$2
tag=$3
dataset=$4
num_tasks=$5
nc_first_task=$6
network=$7
num_epochs=$8
lamb=$9
wu_epochs=${10:-0}
wu_lr=${11:-0.1}
lr=${12:-0.1}
head_init=${13}
stop_at_task=${14:-0}

if [ "${dataset}" = "imagenet_subset_kaggle" ]; then
  clip=1.0
else
  clip=100.0
fi

if [ ${wu_epochs} -gt 0 ]; then
  exp_name="cifar100t${num_tasks}s${nc_first_task}_${tag}_wu_hz_no_wd"
  result_path="results/${tag}/lwf_wu_hz_${lamb}_${seed}"
  python3 src/main_incremental.py \
    --exp-name ${exp_name} \
    --gpu ${gpu} \
    --datasets ${dataset} \
    --num-tasks ${num_tasks} \
    --nc-first-task ${nc_first_task} \
    --network ${network} \
    --use-test-as-val \
    --lr ${lr} \
    --wu-wd 0.0 \
    --clipping ${clip} \
    --nepochs ${num_epochs} \
    --batch-size 128 \
    --seed ${seed} \
    --log disk wandb \
    --results-path ${result_path} \
    --tags ${tag} \
    --scheduler-milestones \
    --cm \
    --approach lwf \
    --taskwise-kd \
    --stop-at-task ${stop_at_task} \
    --lamb ${lamb} \
    --wu-nepochs ${wu_epochs} \
    --wu-lr ${wu_lr} \
    --wu-fix-bn \
    --wu-scheduler cosine \
    --head-init-mode ${head_init}
else
  exp_name="cifar100t${num_tasks}s${nc_first_task}_${tag}_hz_no_wd"
  result_path="results/${tag}/lwf_hz_${lamb}_${seed}"
  python3 src/main_incremental.py \
    --exp-name ${exp_name} \
    --gpu ${gpu} \
    --datasets ${dataset} \
    --num-tasks ${num_tasks} \
    --nc-first-task ${nc_first_task} \
    --network ${network} \
    --use-test-as-val \
    --lr ${lr} \
    --wu-wd 0.0 \
    --clipping ${clip} \
    --nepochs ${num_epochs} \
    --batch-size 128 \
    --seed ${seed} \
    --log disk wandb \
    --results-path ${result_path} \
    --tags ${tag} \
    --scheduler-milestones \
    --cm \
    --approach lwf \
    --taskwise-kd \
    --stop-at-task ${stop_at_task} \
    --lamb ${lamb} \
    --head-init-mode ${head_init}
fi
