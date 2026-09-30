#!/usr/bin/env bash
set -euo pipefail
export PATH="/root/rlinf_recap/RLinf/.venv/bin:$PATH"
export REPO_PATH=/root/rlinf_recap/RLinf
export PYTHONPATH=/root/rlinf_recap/RLinf
export RAY_TMPDIR=/root/rlinf_recap/logs/ray_tmp
export HF_HOME=/root/.cache/huggingface
export TRANSFORMERS_CACHE=/root/.cache/huggingface
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
mkdir -p "$RAY_TMPDIR"
cd "$REPO_PATH"
exec bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh cfg_rl_openpi
