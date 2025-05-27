#!/usr/bin/env bash
# run_reward_hacking_parallel.sh  <parent_dir>  <dataset_s3_uri>  <model_type>  [gpu_list]
# -----------------------------------------------------------------------------
# Parallel‑serve and run *reward_hacking_pipeline.py* for every model checkpoint
# (sub‑directory) under <parent_dir> using a pool of GPUs. One server per GPU
# is launched on a dedicated port (8000 + 10*i).
#
#   1. Starts *grader_ablation_server.py* with the model from the checkpoint
#      directory (stdout/err → ./logs/ but not spammed to console).
#   2. Executes `reward_hacking_pipeline.py` against the running server with
#      user‑supplied <dataset_s3_uri> and <model_type>.  Model name is derived
#      from the checkpoint directory basename (e.g. step_220 → step_220).
#   3. Shuts the server down after the pipeline finishes.
#
# All intermediate/final Parquet files are written inside the pipeline's own
# `outputs/` directory; nothing is renamed here, so no race conditions.
#
# Requirements: jq, nc (netcat), GNU parallel‑safe bash.
# -----------------------------------------------------------------------------
set -euo pipefail

[[ $# -lt 3 ]] && {
  echo "usage: $0 <parent_dir> <dataset_s3_uri> <model_type:[reasoning|reasoning-summary]> [gpu_list]" >&2
  exit 1
}

###############################################################################
# Args / config                                                                #
###############################################################################
PARENT_DIR=$1                     # directory containing model sub‑dirs
DATASET_S3_URI=$2                 # e.g. s3://my-bucket/reward_hacking/questions.parquet
MODEL_TYPE=$3                     # reasoning | reasoning-summary
IFS=',' read -ra GPUS <<< "${4:-0}"   # comma‑separated GPU IDs (default 0)
NGPUS=${#GPUS[@]}

ROOT=$(pwd -P)                    # absolute path to current dir
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

###############################################################################
# Discover sub‑directories (sorted)                                            #
###############################################################################
mapfile -t DIRS < <(find "$PARENT_DIR" -maxdepth 1 -mindepth 1 -type d | sort)

###############################################################################
# Worker – one subshell per GPU                                                #
###############################################################################
for IDX in "${!GPUS[@]}"; do (
  GPU_ID="${GPUS[$IDX]}"
  PORT=$((8000 + 10*IDX))

  for ((i=IDX; i<${#DIRS[@]}; i+=NGPUS)); do
    DIR="${DIRS[$i]%/}"
    CKPT_NAME="${DIR##*/}"          # use full directory name as model_name

    PIPE_LOG="$LOG_DIR/pipeline_${CKPT_NAME}_gpu${GPU_ID}.log"
    SRV_LOG="$LOG_DIR/server_${CKPT_NAME}_gpu${GPU_ID}.log"

    echo "[$(date +%H:%M:%S)] GPU $GPU_ID → $DIR (port $PORT)"

    ######################
    # 1. start the server
    ######################
    VLLM_USE_V1=0 \
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    PORT=$PORT \
    MODEL_PATH="$DIR" \
    python -u "$ROOT/sft/grader_ablation_server.py" \
      > "$SRV_LOG" 2>&1 &
    SERVER_PID=$!

    # wait (up to 180s) for server to open the port; else abort this dir
    for _ in {1..180}; do
      nc -z localhost "$PORT" 2>/dev/null && break
      sleep 1
    done
    if ! nc -z localhost "$PORT" 2>/dev/null; then
      echo "[WARN] Server for $CKPT_NAME on GPU $GPU_ID failed to start. See $SRV_LOG" >&2
      kill "$SERVER_PID" 2>/dev/null || true
      wait "$SERVER_PID" 2>/dev/null || true
      continue   # skip to next DIR
    fi

    echo "[$(date +%H:%M:%S)] GPU $GPU_ID → server ready"

    ######################
    # 2. run the pipeline
    ######################
    # The sampler picks up PORT via environment; no GPU needed for pipeline.
    PORT=$PORT \
    python -u "$ROOT/reward_hacking_pipeline.py" \
      --dataset_s3_key "$DATASET_S3_URI" \
      --model_name "$CKPT_NAME" \
      --model_type "$MODEL_TYPE" \
      > "$PIPE_LOG" 2>&1

    ######################
    # 3. shutdown server
    ######################
    kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true
    echo "[$(date +%H:%M:%S)] GPU $GPU_ID → stopped server"
  done
)& done
wait  # wait for all GPU workers

echo -e "\n✓ All done. Logs in $LOG_DIR; pipeline outputs live under ./outputs/"
