task_name=${1}
setting=${2}
max_num=${3}
camera_type=${4}
save_path=${5:-"/workspace/embolab/data"}

python utils/generate_episode_instructions.py $task_name $setting $max_num $camera_type $save_path
