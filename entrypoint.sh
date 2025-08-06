#!/bin/bash

get_json_value() {
    python3 -c "import sys, json; print(json.load(open('/workspace/embolab/params/build_task.json'))$1)"
}

TASK_NAME=$(get_json_value "['task_name']")
CAMERA_TYPE=$(get_json_value "['data_generation']['camera_type']")
EXPERT_DATA_NUM=$(get_json_value "['data_generation']['episode']")
GPU_ID=$(get_json_value "['gpu_id']")
SAVE_PATH=$(get_json_value "['data_generation']['output_path']")
LOG_DIR=$(get_json_value "['data_generation']['log_path']")

echo "Task Name: $TASK_NAME"
echo "Camera Type: $CAMERA_TYPE"
echo "Expert Data Number: $EXPERT_DATA_NUM"
echo "GPU ID: $GPU_ID"
echo "Save Path: $SAVE_PATH"

mkdir -p "$LOG_DIR"
DATE_STR=$(date +"%Y-%m-%d_%H-%M-%S")
UUID=$(uuidgen)
LOG_FILE="${LOG_DIR}/${TASK_NAME}_${CAMERA_TYPE}_${EXPERT_DATA_NUM}_${DATE_STR}_${UUID}.log"
TASK_CONFIG=demo_randomized
#source ~/miniconda3/bin/activate RoboTwin
source /opt/conda/bin/activate RoboTwin


cd "$(dirname "$0")"
{
    echo "=== Data Generating Started at $(date) ==="
    bash collect_data.sh "${TASK_NAME}" "${TASK_CONFIG}"  "${GPU_ID}"  "${CAMERA_TYPE}" "${EXPERT_DATA_NUM}"  
    EXIT_CODE=$?
    echo "=== Data Generation Finished at $(date) ==="
    echo "Exit Code: $EXIT_CODE"
    exit $EXIT_CODE
} | tee -a "$LOG_FILE"

echo "Log file saved to: $LOG_FILE"


#bash collect_data.sh beat_block_hammer demo_randomized 0 L515 1 