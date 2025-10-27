#!/bin/bash

policy_name=DP3
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4} # both policy and RoboTwin scen
gpu_id=${5}
checkpoint_num=${6} # default to 10 if not provided
checkpoint_path=${7:-"/workspace/embolab/checkpoints"}

export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --gpu_id ${gpu_id} \
    --checkpoint_path ${checkpoint_path} \
    --checkpoint_num ${checkpoint_num}
    