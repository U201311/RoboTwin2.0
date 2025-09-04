#!/bin/bash

get_json_value() {
    python3 -c "import sys, json; print(json.load(open('/workspace/embolab/params/build_task.json'))$1)"
}

TASK_NAME=$(get_json_value "['evaluation']['task_name']")
CAMERA_TYPE=$(get_json_value "['evaluation']['camera_type']")
SEED=$(get_json_value "['evaluation']['seed']")
TASK_CONFIG=$(get_json_value "['evaluation']['task_config']")
GPU_ID=$(get_json_value "['gpu_id']")
CHECKPOINT_PATH=$(get_json_value "['evaluation']['checkpoint_path']")
LOG_DIR=$(get_json_value "['evaluation']['log_path']")
OUTPUT_DIR=$(get_json_value "['evaluation']['output_path']")
mkdir -p "$LOG_DIR"
model_name="moe_demo_randomized"
train_config_name="pi0_multi_res_moe_aloha_robotwin_full"
CHECKPOINT_ID=$(get_json_value "['evaluation']['checkpoint_id']")

DATE_STR=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="${LOG_DIR}/${TASK_NAME}_${CAMERA_TYPE}_${SEED}_${DATE_STR}.log"


#补充echo
echo "Task Name: ${TASK_NAME}"
echo "Camera Type: ${CAMERA_TYPE}"
echo "GPU ID: ${GPU_ID}"
echo "Seed: ${SEED}"
echo "Checkpoint Path: ${CHECKPOINT_PATH}"
echo "Log Directory: ${LOG_DIR}"

# Try to deactivate conda environment safely
# conda deactivate

# 切换到 DP3 工作目录
cd "$(dirname "$0")/policy/pi0"
source .venv/bin/activate



# 将所有输出重定向到日志文件
exec > >(tee -a "${LOG_FILE}") 2>&1
#python scripts/pkl2zarr_dp3.py ${TASK_NAME} ${CAMERA_TYPE} ${EXPERT_DATA_NUM} --load_dir ${FULL_DATA_PATH} --save_dir /workspace/3D-Diffusion-Policy/data/${TASK_NAME}_${CAMERA_TYPE}_${EXPERT_DATA_NUM}.zarr
#python scripts/pkl2zarr_dp3.py dual_shoes_place D435 1 --load_dir /workspace/3D-Diffusion-Policy/data2/dual_shoes_place_D435_pkl --save_dir /workspace/3D-Diffusion-Policy/data/dual_shoes_place_D435_1.zarr
#python scripts/pkl2zarr_dp3.py blocks_stack_hard D435 1 --load_dir /workspace/3D-Diffusion-Policy/data2/blocks_stack_hard_D435_pkl --save_dir /workspace/3D-Diffusion-Policy/data/blocks_stack_hard_D435_1.zarr

echo "Starting eval for task: ${TASK_NAME} with camera type: ${CAMERA_TYPE} "
bash eval.sh ${TASK_NAME} ${TASK_CONFIG} ${train_config_name} ${model_name} ${SEED} ${GPU_ID} ${CHECKPOINT_ID}
#bash eval.sh beat_block_hammer demo_randomized pi0_multi_res_moe_aloha_robotwin_full moe_demo_randomized 0 4,5,6,7 80000