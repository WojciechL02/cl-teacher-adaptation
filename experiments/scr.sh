#!/bin/bash

# set -e

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
exemplars=${12:-20}
temperature=${13:-0.1}
batch_size=${14:-128}

exp_name="t${num_tasks}s${nc_first_task}_hz_m:${exemplars}"
result_path="results/${tag}/scr_hz_${seed}"
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
    --batch-size ${batch_size} \
    --seed ${seed} \
    --log disk wandb \
    --results-path ${result_path} \
    --tags ${tag} \
    --scheduler-milestones \
    --stop-at-task ${stop_at_task} \
    --approach scr \
    --temperature ${temperature} \
    --num-exemplars ${exemplars} \
    --head-init-mode ${head_init} \
    --classifier nmc \
    --extra-aug simclr_cifar
