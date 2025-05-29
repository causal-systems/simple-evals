#!/usr/bin/env bash
# run_all_checkpoints.sh — simple GPU-aware semaphore

set -euo pipefail
shopt -s nullglob        # silently ignore missing matches

CKPT_ROOT=${CKPT_ROOT:-}
LOG_DIR="./logs"; mkdir -p "$LOG_DIR"

# Detect GPU indices (e.g. 0 1 2 …)
readarray -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
(( ${#GPUS[@]} )) || { echo "No GPUs detected." >&2; exit 1; }
echo "Found ${#GPUS[@]} GPU(s): ${GPUS[*]}"

declare -A GPU_PID       # gpu-idx → child-PID  (empty = free)

# Block until a GPU is free, then print its index
wait_for_gpu() {
  while true; do
    for g in "${GPUS[@]}"; do
      pid="${GPU_PID[$g]-}"
      if [[ -z $pid ]]; then             # never used
        echo "$g"; return
      elif ! kill -0 "$pid" 2>/dev/null; then  # finished
        unset 'GPU_PID[$g]'
        echo "$g"; return
      fi
    done
    sleep 1
  done
}

for ckpt in "$CKPT_ROOT"/*; do
  [[ -d $ckpt ]] || continue
  gpu=$(wait_for_gpu)
  name=$(basename "$ckpt")
  echo "[$(date '+%F %T')] START $ckpt on GPU $gpu"

  (
    CUDA_VISIBLE_DEVICES=$gpu ./run_reward_hacking_pipeline.sh "$ckpt" \
      > "$LOG_DIR/$name.log" 2>&1
    echo "[$(date '+%F %T')] DONE  $name on GPU $gpu (exit $?)"
  ) &

  GPU_PID[$gpu]=$!        # mark GPU busy with child PID
done

wait
echo "All checkpoints finished."

