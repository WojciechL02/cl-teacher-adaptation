#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

# set -e

# eval "$(conda shell.bash hook)"
# conda activate FACIL

num_tasks=10
nc_first_task=10
# nc_per_task="2 2 2 2 2 20 20 20 10 20"
stop_at_task=5  # default = 0
dataset=cifar100_icarl
network=resnet18
tag=proj_architecture  # experiment name

num_epochs=100
bsz=128
lr=0.1
head_init=zeros
seed=0
exemplars=2000

exp_name="t${num_tasks}s20_hz_m:${exemplars}_full"
result_path="results/${tag}/ft_${seed}_full"
python3 src/main_incremental.py \
    --exp-name ${exp_name} \
    --gpu 0 \
    --datasets ${dataset} \
    --num-tasks ${num_tasks} \
    --nc-first-task ${nc_first_task} \
    --network ${network} \
    --use-test-as-val \
    --lr ${lr} \
    --nepochs ${num_epochs} \
    --batch-size ${bsz} \
    --seed ${seed} \
    --log wandb \
    --cm \
    --cont-eval \
    --results-path ${result_path} \
    --tags ${tag} \
    --stop-at-task ${stop_at_task} \
    --approach finetuning \
    --num-exemplars ${exemplars} \
    --head-init-mode ${head_init} \
    --scheduler-type linear \
    --classifier linear \
    --projector_type full
