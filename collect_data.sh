#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}
camera_type=${4}
episode=${5}
#save_path=${6:-"/workspace/robotwin_generation/data"}
save_path="/workspace/embolab/data"


./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py $task_name $task_config --camera_type $camera_type --episode $episode --gpu_id $gpu_id --save_path $save_path

