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
lr=${9:-0.1}
head_init=${10}
stop_at_task=${11:-0}
lamb=${12:-1}
update_prototypes=${13:-0}


if [ ${update_prototypes} -gt 0 ]; then
    exp_name="cifar100t${num_tasks}s${nc_first_task}_${tag}_hz_up:${update_prototypes}"
    result_path="results/${tag}/msp_nmc_hz_${seed}"
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
    --cm \
    --stop-at-task ${stop_at_task} \
    --approach msp_nmc \
    --lamb ${lamb} \
    --num-exemplars 2000 \
    --head-init-mode ${head_init} \
    --update_prototypes
else
    exp_name="cifar100t${num_tasks}s${nc_first_task}_${tag}_hz_up:${update_prototypes}"
    result_path="results/${tag}/msp_nmc_hz_${seed}"
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
    --cm \
    --stop-at-task ${stop_at_task} \
    --approach msp_nmc \
    --lamb ${lamb} \
    --num-exemplars 2000 \
    --head-init-mode ${head_init}
fi
