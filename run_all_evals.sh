#!/usr/bin/env bash
# run_evals_parallel.sh  <parent_dir>  [gpu_list]
# -------------------------------------------------
# Parallel‑serve and evaluate every model sub‑directory under <parent_dir> using a
# pool of GPUs. Each GPU (worker index) gets its dedicated port (8000 + 10*i).
#   * The model server is started with stdout/err logged (./logs/) but not
#     spammed to the console.
#   * simple‑evals honours OUTPUT_PATH, so each run writes directly to
#     simple_evals_output_<step>.json — no rename races.
#   * Results are aggregated into JSONL & CSV after all workers finish.
#
# Requirements: jq, nc (netcat).
# -----------------------------------------------------------------------------
set -euo pipefail

[[ $# -lt 1 ]] && { echo "usage: $0 <parent_dir> [gpu_list]"; exit 1; }

###############################################################################
# Args / config
###############################################################################
PARENT_DIR=$1                       # directory containing model sub-dirs
IFS=',' read -ra GPUS <<< "${2:-0}"     # comma‑separated GPU IDs (default 0)
NGPUS=${#GPUS[@]}

ROOT=$(pwd -P)                      # absolute path to current dir
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

ALL_JSONL="simple_evals_all.jsonl"
ALL_CSV="simple_evals_all.csv"
: > "$ALL_JSONL"                  # truncate / create

###############################################################################
# Discover sub‑directories (sorted)
###############################################################################
mapfile -t DIRS < <(find "$PARENT_DIR" -maxdepth 1 -mindepth 1 -type d | sort)

###############################################################################
# Worker – one subshell per GPU
###############################################################################
for IDX in "${!GPUS[@]}"; do (
  GPU_ID="${GPUS[$IDX]}"
  PORT=$((8000 + 10*IDX))

  for ((i=IDX; i<${#DIRS[@]}; i+=NGPUS)); do
    DIR="${DIRS[$i]%/}"
    STEP="${DIR##*_}"                # e.g. step_220 → 220

    OUT_JSON="$ROOT/simple_evals_output_${STEP}.json"
    SRV_LOG="$LOG_DIR/server_step${STEP}_gpu${GPU_ID}.log"
    EVAL_LOG="$LOG_DIR/eval_step${STEP}_gpu${GPU_ID}.log"

    echo "[$(date +%H:%M:%S)] GPU $GPU_ID → $DIR (step $STEP, port $PORT)"

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

    # wait (up to 120s) for server to open the port; else abort this step
    for _ in {1..120}; do
      nc -z localhost "$PORT" 2>/dev/null && break
      sleep 1
    done
    if ! nc -z localhost "$PORT" 2>/dev/null; then
      echo "[WARN] Server for step $STEP on GPU $GPU_ID failed to start. See $SRV_LOG" >&2
      kill "$SERVER_PID" 2>/dev/null || true
      wait "$SERVER_PID" 2>/dev/null || true
      continue   # skip to next DIR
    fi

    ######################
    # 2. run the evals
    ######################
    PORT=$PORT \
    EVALS="gpqa;math" \
    OUTPUT_PATH="$OUT_JSON" \
    GRADER_ENDPOINT=https://xu9earqqqeaf9l-8000.proxy.runpod.net/grade \
    python -m simple-evals.simple_evals --model grader_ablation_3b \
      > "$EVAL_LOG" 2>&1

    ######################
    # 3. shutdown server
    ######################
    kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true
  done
)& done
wait  # wait for all GPU workers

###############################################################################
# Collate individual JSONs → JSONL & CSV
###############################################################################
for J in simple_evals_output_*.json; do
  STEP="${J##*_}"; STEP="${STEP%.json}"
  jq -c --arg step "$STEP" '{step:($step|tonumber),results:.}' "$J" >> "$ALL_JSONL"
done
jq -r '[.step,(.results|tojson)]|@csv' "$ALL_JSONL" > "$ALL_CSV"

echo -e "\n✓ All done. Outputs reside in:"
printf '  %s\n' simple_evals_output_*.json "$ALL_JSONL" "$ALL_CSV"
echo "Logs in $LOG_DIR"
