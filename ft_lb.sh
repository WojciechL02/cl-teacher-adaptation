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
lr=${13:-0.1}
head_init=${14}
stop_at_task=${15:-0}
classifier=${16}
h=${17}
name=${18}

exp_name="t${num_tasks}s20_hz_m:${exemplars}_${name}"
result_path="results/${tag}/ft_lb_hz_${seed}"
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
    --batch-size ${bsz} \
    --seed ${seed} \
    --log disk wandb \
    --results-path ${result_path} \
    --tags ${tag} \
    --cm \
    --scheduler-type linear \
    --stop-at-task ${stop_at_task} \
    --approach ft_lb \
    --num-exemplars ${exemplars} \
    --head-init-mode ${head_init} \
    --classifier ${classifier} \
    --lamb ${lamb} \
    --h ${h}
