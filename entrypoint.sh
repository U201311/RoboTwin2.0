#!/bin/bash

get_json_value() {
    python3 -c "import sys, json; print(json.load(open('/workspace/embolab/params/build_task.json'))$1)"
}

TASK_NAME=$(get_json_value "['task_name']")
CAMERA_TYPE=$(get_json_value "['train']['camera_type']")
EXPERT_DATA_NUM=$(get_json_value "['train']['episode']")
EXPERT_DATA_NUM=$((EXPERT_DATA_NUM))
INPUT_DATA_PATH=$(get_json_value "['train']['input_data_path']")
GPU_ID=$(get_json_value "['gpu_id']")
SEED=$(get_json_value "['train']['seed']")
EPOCHS=$(get_json_value "['train']['epochs']")
CHECKPOINT_PATH=$(get_json_value "['train']['checkpoint_path']")
LOG_DIR=$(get_json_value "['train']['log_path']")

mkdir -p "$LOG_DIR"

DATE_STR=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="${LOG_DIR}/${TASK_NAME}_${CAMERA_TYPE}_${EXPERT_DATA_NUM}_${DATE_STR}.log"


#补充echo
echo "Task Name: ${TASK_NAME}"
echo "Camera Type: ${CAMERA_TYPE}"
echo "Expert Data Number: ${EXPERT_DATA_NUM}"
echo "Input Data Path: ${INPUT_DATA_PATH}"      
echo "GPU ID: ${GPU_ID}"
echo "Seed: ${SEED}"
echo "Epoch: ${EPOCHS}"
echo "Checkpoint Path: ${CHECKPOINT_PATH}"
echo "Log Directory: ${LOG_DIR}"


# 切换到 DP3 工作目录
cd "$(dirname "$0")/policy/DP3"
# 将所有输出重定向到日志文件
exec > >(tee -a "${LOG_FILE}") 2>&1
source /opt/conda/bin/activate RoboTwin

echo "Starting data transfer for task: ${TASK_NAME} with camera type: ${CAMERA_TYPE} and expert data number: ${EXPERT_DATA_NUM}"
# 拼凑完整的数据路径
bash process_data.sh ${TASK_NAME} demo_randomized ${EXPERT_DATA_NUM} ${CAMERA_TYPE}
#python scripts/pkl2zarr_dp3.py ${TASK_NAME} ${CAMERA_TYPE} ${EXPERT_DATA_NUM} --load_dir ${FULL_DATA_PATH} --save_dir /workspace/3D-Diffusion-Policy/data/${TASK_NAME}_${CAMERA_TYPE}_${EXPERT_DATA_NUM}.zarr
#python scripts/pkl2zarr_dp3.py dual_shoes_place D435 1 --load_dir /workspace/3D-Diffusion-Policy/data2/dual_shoes_place_D435_pkl --save_dir /workspace/3D-Diffusion-Policy/data/dual_shoes_place_D435_1.zarr
#python scripts/pkl2zarr_dp3.py blocks_stack_hard D435 1 --load_dir /workspace/3D-Diffusion-Policy/data2/blocks_stack_hard_D435_pkl --save_dir /workspace/3D-Diffusion-Policy/data/blocks_stack_hard_D435_1.zarr

echo "Starting training for task: ${TASK_NAME} with camera type: ${CAMERA_TYPE} and expert data number: ${EXPERT_DATA_NUM}"
bash train.sh ${TASK_NAME}  demo_randomized  ${EXPERT_DATA_NUM} ${SEED} ${GPU_ID}  ${EPOCHS} 
#bash train.sh beat_block_hammer demo_randomized 1 0 0 50 
#bash train.sh dual_shoes_place D435 1 5000 0 5 /workspace/3D-Diffusion-Policy/checkpoints
