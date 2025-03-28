#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

# set -e

# eval "$(conda shell.bash hook)"
# conda activate FACIL

num_tasks=10
nc_first_task=10
stop_at_task=3  # default = 0
dataset=cifar100_icarl
network=resnet18
tag=lie_bracket_4  # experiment name

num_epochs=100
bsz=128
lr=0.1
head_init=zeros
seed=0
classifier=linear

exp_name="t${num_tasks}s20_hz_m:${exemplars}"
result_path="results/${tag}/lwf_lb_hz_${seed}"
python3 src/main_incremental.py \
    --exp-name ${exp_name} \
    --gpu 1 \
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
    --cont-eval \
    --results-path ${result_path} \
    --tags ${tag} \
    --cm \
    --scheduler-type linear \
    --stop-at-task ${stop_at_task} \
    --approach lwf_lb \
    --taskwise-kd \
    --head-init-mode ${head_init} \
    --classifier linear \
    --lamb 1 \
    --optimizer-type lb \
    --ha 0.1 \
    --classifier ${classifier}
