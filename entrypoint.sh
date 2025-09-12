#!/bin/bash

# 错误处理：遇到错误时停止执行
set -e

# 检查必需文件是否存在
if [ ! -f "/workspace/embolab/params/build_task.json" ]; then
    echo "错误: 未找到配置文件 /workspace/embolab/params/build_task.json"
    exit 1
fi

get_json_value() {
    python3 -c "import sys, json; print(json.load(open('/workspace/embolab/params/build_task.json'))$1)"
}

INPUT_DATA_PATH=$(get_json_value "['train']['input_data_path']")
GPU_ID=$(get_json_value "['gpu_id']")
EPOCHS=$(get_json_value "['train']['epochs']")
CHECKPOINT_PATH=$(get_json_value "['train']['checkpoint_path']")
LOG_DIR=$(get_json_value "['train']['log_path']")

mkdir -p "$LOG_DIR"

DATE_STR=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="${LOG_DIR}/${DATE_STR}.log"


echo "Input Data Path: ${INPUT_DATA_PATH}"      
echo "GPU ID: ${GPU_ID}"
echo "Epoch: ${EPOCHS}"
echo "Checkpoint Path: ${CHECKPOINT_PATH}"
echo "Log Directory: ${LOG_DIR}"

# 切换到 DP3 工作目录
cd "$(dirname "$0")/policy/DP3"
# 将所有输出重定向到日志文件
exec > >(tee -a "${LOG_FILE}") 2>&1
source /opt/conda/bin/activate RoboTwin

TASK_COUNT=$(python3 -c "import sys, json; data = json.load(open('/workspace/embolab/params/build_task.json')); print(len(data['train']['task_list']))")

for ((i=0; i<$TASK_COUNT; i++)); do
    echo "处理任务 $((i+1))/$TASK_COUNT"
    
    # 提取每个任务的参数
    TASK_NAME=$(python3 -c "import sys, json; data = json.load(open('/workspace/embolab/params/build_task.json')); print(data['train']['task_list'][$i]['task_name'])")
    TASK_CONFIG=$(python3 -c "import sys, json; data = json.load(open('/workspace/embolab/params/build_task.json')); print(data['train']['task_list'][$i]['task_config'])")
    CAMERA_TYPE=$(python3 -c "import sys, json; data = json.load(open('/workspace/embolab/params/build_task.json')); print(data['train']['task_list'][$i]['camera_type'])")
    EXPERT_DATA_NUM_TASK=$(python3 -c "import sys, json; data = json.load(open('/workspace/embolab/params/build_task.json')); print(data['train']['task_list'][$i]['expert_data_num'])")
    
    echo "任务名称: $TASK_NAME"
    echo "任务配置: $TASK_CONFIG"
    echo "摄像头类型: $CAMERA_TYPE"
    echo "专家数据数量: $EXPERT_DATA_NUM_TASK"
    
    #打印开始处理数据
    echo "开始处理数据..., 任务名称: ${TASK_NAME}"
    bash process_data.sh ${TASK_NAME} ${TASK_CONFIG} ${EXPERT_DATA_NUM_TASK} ${CAMERA_TYPE}

done

python scripts/merge_data.py

TASK_CONFIG="demo_randomized"
EXPERT_DATA_NUM=100
SEED=5000
TASK_NAME="click_alarmclock"

#echo "Starting data transfer for task: ${TASK_NAME} with camera type: ${CAMERA_TYPE} and expert data number: ${EXPERT_DATA_NUM}"
# 拼凑完整的数据路径
# bash process_data.sh ${TASK_NAME} ${TASK_CONFIG} ${EXPERT_DATA_NUM} ${CAMERA_TYPE}
#python scripts/pkl2zarr_dp3.py ${TASK_NAME} ${CAMERA_TYPE} ${EXPERT_DATA_NUM} --load_dir ${FULL_DATA_PATH} --save_dir /workspace/3D-Diffusion-Policy/data/${TASK_NAME}_${CAMERA_TYPE}_${EXPERT_DATA_NUM}.zarr
#python scripts/pkl2zarr_dp3.py dual_shoes_place D435 1 --load_dir /workspace/3D-Diffusion-Policy/data2/dual_shoes_place_D435_pkl --save_dir /workspace/3D-Diffusion-Policy/data/dual_shoes_place_D435_1.zarr
#python scripts/pkl2zarr_dp3.py blocks_stack_hard D435 1 --load_dir /workspace/3D-Diffusion-Policy/data2/blocks_stack_hard_D435_pkl --save_dir /workspace/3D-Diffusion-Policy/data/blocks_stack_hard_D435_1.zarr

echo "Starting training for multiple tasks..."
bash train.sh ${TASK_NAME}  ${TASK_CONFIG}  ${EXPERT_DATA_NUM} ${SEED} ${GPU_ID}  ${EPOCHS} 
#bash train.sh click_alarmclock demo_randomized 100 5000 2 1000


rm -rf /workspace/robotwin_generation/policy/DP3/data