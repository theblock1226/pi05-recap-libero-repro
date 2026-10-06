#!/usr/bin/env bash
# Invoked by watch_cfg_post_24000.sh after global_step_30000 is exported.
# Filename is unchanged so the already-running watcher still reaches this file.
# Contents are no longer Libero-10 x10. They are:
#   LIBERO-10 Task 0 x 50 vanilla, then Task 0 x 50 CFG positive,
#   and the CFG run continues until 10 success videos or 200 trials.
set -euo pipefail
export REPO_PATH=/root/rlinf_recap/RLinf
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export HF_HOME=/root/.cache/huggingface

cd "$REPO_PATH"
exec /root/rlinf_recap/RLinf/.venv/bin/python -u /root/rlinf_recap/logs/eval_cfg30000_task0.py
