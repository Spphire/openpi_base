#!/bin/bash
# Online DAgger training script for OpenPi
#
# Usage:
#   ./train_online_dagger.sh
#
# Configuration variables below should be modified according to your setup.

# Training config name (should match an existing config in config.py)
CONFIG_NAME=pi05_iPhoneSingle_blocks_100

# Experiment name
EXP_NAME=blocks_100_dagger_run1

# Examples: "./checkpoints/<config>/<exp>/<step>/params" or "gs://openpi-assets/checkpoints/<model>/params"
# Leave empty to use the default weight_loader from base config
INIT_CHECKPOINT=None

# Offline dataset repo ID (the base dataset)
OFFLINE_REPO_ID=flexiv/blocks_100

# Online dataset repo ID (will be created for online data)
ONLINE_REPO_ID=flexiv/blocks_100_online

# Data cloud settings
DATACLOUD_ENDPOINT=http://8.153.94.10:8080
DAGGER_IDENTIFIER=blocks_100_dagger_debug

# Robot configuration
ROBOT_TYPE=single_iphone_flexiv
FPS=10
TASK_DESCRIPTION="Stack the blocks in order."

# zarr configs
ACTION_TYPE=left_arm_6DOF_gripper_width
TEMPORAL_DOWNSAMPLE_RATIO=0
EPISODE_CLIP_HEAD_SECONDS=0.0
EPISODE_CLIP_TAIL_SECONDS=0.0
GRIPPER_WIDTH_BIAS=0.0
GRIPPER_WIDTH_SCALE=1.0

# Adaptive sampling parameters
FETCH_INTERVAL=1
WINDOW_SIZE=200
BOOST_FACTOR=1.5
MIN_ONLINE_RATIO=0.2
MAX_ONLINE_RATIO=0.8
INITIAL_ONLINE_WEIGHT=0.5

# Checkpoint sync settings for DAgger backend
# Set INFERENCE_SERVER_URL to enable checkpoint sync (e.g., "http://localhost:8080")
# Leave empty to disable checkpoint sync
INFERENCE_SERVER_URL=""
SYNC_INTERVAL=1000  # Sync checkpoint every N steps
BASE_WORKSPACE_CONFIG=""  # Leave empty to use CONFIG_NAME
BASE_TASK_CONFIG=""  # Leave empty to use TASK_DESCRIPTION
SYNC_TIMEOUT=120
SYNC_RETRIES=3

# GPU settings
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
USED_GPUS=0,1,2,3

# Run training
CUDA_VISIBLE_DEVICES=${USED_GPUS} python scripts/train_online_dagger.py \
    \
    `# === Training config ===` \
    --config-name=${CONFIG_NAME} \
    --exp-name=${EXP_NAME} \
    --init-checkpoint=${INIT_CHECKPOINT} \
    --overwrite \
    \
    `# === Data cloud settings ===` \
    --online-repo-id=${ONLINE_REPO_ID} \
    --datacloud-endpoint=${DATACLOUD_ENDPOINT} \
    --identifier=${DAGGER_IDENTIFIER} \
    --fetch-interval=${FETCH_INTERVAL} \
    \
    `# === Robot configuration ===` \
    --robot-type=${ROBOT_TYPE} \
    --fps=${FPS} \
    --task-description="${TASK_DESCRIPTION}" \
    \
    `# === Zarr configs ===` \
    --action-type=${ACTION_TYPE} \
    --temporal-downsample-ratio=${TEMPORAL_DOWNSAMPLE_RATIO} \
    --use-dino \
    --episode-clip-head-seconds=${EPISODE_CLIP_HEAD_SECONDS} \
    --episode-clip-tail-seconds=${EPISODE_CLIP_TAIL_SECONDS} \
    --gripper-width-bias=${GRIPPER_WIDTH_BIAS} \
    --gripper-width-scale=${GRIPPER_WIDTH_SCALE} \
    \
    `# === Adaptive sampling parameters ===` \
    --window-size=${WINDOW_SIZE} \
    --boost-factor=${BOOST_FACTOR} \
    --min-online-ratio=${MIN_ONLINE_RATIO} \
    --max-online-ratio=${MAX_ONLINE_RATIO} \
    --initial-online-weight=${INITIAL_ONLINE_WEIGHT} \
    \
    `# === Checkpoint sync settings ===` \
    --inference-server-url=${INFERENCE_SERVER_URL} \
    --sync-interval=${SYNC_INTERVAL} \
    --base-workspace-config=${BASE_WORKSPACE_CONFIG} \
    --base-task-config=${BASE_TASK_CONFIG} \
    --sync-timeout=${SYNC_TIMEOUT} \
    --sync-retries=${SYNC_RETRIES}
