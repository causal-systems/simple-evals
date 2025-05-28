#!/usr/bin/env bash
# Sequentially run “model”, “grader”, “verifier” for every checkpoint directory
# under $CHECKPOINT_ROOT.  At most one checkpoint runs on each visible GPU.

set -euo pipefail

###############################################################################
# CONFIG  (override via env vars if desired)
###############################################################################
HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
HF_TOKEN=${HF_TOKEN:-}               # optional – warns if missing
VLLM_USE_V1=0

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/workspace/3b_trained_against_verifier_with_summary}
DATASET_URI=${DATASET_URI:-s3://grteambucket/genverify/grader_ablation/grader_eval.parquet}
MODEL_TYPE=reasoning-summary
###############################################################################

[[ -z $HF_TOKEN ]] && echo "⚠️  HF_TOKEN not set – proceeding without it" >&2

for stage in model grader verifier; do
  python -m simple-evals.reward_hacking_pipeline \
    --dataset-s3-uri "$DATASET_URI" \
    --checkpoint-dir "$1" \
    --model-type     "$MODEL_TYPE" \
    --stage          "$stage"
done
