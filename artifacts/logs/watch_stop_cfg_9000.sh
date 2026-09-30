#!/usr/bin/env bash
# Wait for complete global_step_9000, then stop CFG training.
# Does NOT delete global_step_6000.
set -u
CKDIR=/root/rlinf_recap/RLinf/logs/cfg_rl/cfg_rl_openpi-20260929-00:58:47/cfg_sft/checkpoints
REF=$CKDIR/global_step_6000
TARGET=$CKDIR/global_step_9000
PIDFILE=/root/rlinf_recap/logs/step4.pid
WATCHLOG=/root/rlinf_recap/logs/step4_stop_at_9000.log
LOG=/root/rlinf_recap/RLinf/logs/cfg_rl/cfg_rl_openpi-20260929-00:58:47/run_cfg_rl.log

ref_size=$(du -sb "$REF" 2>/dev/null | awk '{print $1}')
if [[ -z "${ref_size:-}" ]]; then
  ref_size=30000000000
fi
echo "$(date -Is) watch start (no delete 6000) ref_size=$ref_size" | tee -a "$WATCHLOG"

is_complete() {
  local d="$1"
  [[ -d "$d/actor/dcp_checkpoint" ]] || return 1
  [[ -f "$d/actor/model_state_dict/full_weights.pt" ]] || return 1
  local sz
  sz=$(du -sb "$d" 2>/dev/null | awk '{print $1}')
  python3 -c "import sys; sys.exit(0 if int('$sz') >= int(float('$ref_size')*0.8) else 1)"
}

get_step() {
  python3 - <<'PY'
from pathlib import Path
import re
t=Path('/root/rlinf_recap/RLinf/logs/cfg_rl/cfg_rl_openpi-20260929-00:58:47/run_cfg_rl.log').read_bytes().decode('utf-8','replace')
last=0
for c in re.split(r'[\r\n]', t):
  if 'train/loss=' in c:
    m=re.search(r'(\d+)/367920000', c)
    if m: last=int(m.group(1))
print(last)
PY
}

for i in $(seq 1 240); do
  step=$(get_step)
  if [[ -d "$TARGET" ]] && is_complete "$TARGET"; then
    echo "$(date -Is) COMPLETE step=$step size=$(du -sh "$TARGET" | awk '{print $1}')" | tee -a "$WATCHLOG"
    break
  fi
  if [[ -d "$TARGET" ]]; then
    echo "$(date -Is) saving... step=$step partial=$(du -sh "$TARGET" 2>/dev/null | awk '{print $1}')" | tee -a "$WATCHLOG"
  else
    echo "$(date -Is) waiting step=$step" | tee -a "$WATCHLOG"
  fi
  sleep 15
done

if ! is_complete "$TARGET"; then
  echo "$(date -Is) FAILED: ckpt 9000 not complete" | tee -a "$WATCHLOG"
  exit 1
fi

MAIN=$(cat "$PIDFILE" 2>/dev/null || true)
echo "$(date -Is) stopping train (keep global_step_6000) main=$MAIN" | tee -a "$WATCHLOG"
pkill -INT -f 'train_cfg.py.*cfg_rl_openpi' 2>/dev/null || true
[[ -n "${MAIN:-}" ]] && kill -INT "$MAIN" 2>/dev/null || true
sleep 10
if pgrep -f 'train_cfg.py.*cfg_rl_openpi' >/dev/null; then
  echo "$(date -Is) SIGTERM" | tee -a "$WATCHLOG"
  pkill -TERM -f 'train_cfg.py.*cfg_rl_openpi' 2>/dev/null || true
  [[ -n "${MAIN:-}" ]] && kill -TERM "$MAIN" 2>/dev/null || true
  sleep 6
fi
if pgrep -f 'train_cfg.py.*cfg_rl_openpi' >/dev/null; then
  echo "$(date -Is) SIGKILL" | tee -a "$WATCHLOG"
  pkill -KILL -f 'train_cfg.py.*cfg_rl_openpi' 2>/dev/null || true
fi
sleep 3

if ! is_complete "$TARGET"; then
  echo "$(date -Is) ERROR: 9000 incomplete after stop" | tee -a "$WATCHLOG"
  exit 2
fi

echo "$(date -Is) 9000 OK $(du -sh "$TARGET")" | tee -a "$WATCHLOG"
echo "$(date -Is) kept ckpts: $(ls "$CKDIR")" | tee -a "$WATCHLOG"
df -h / | tail -1 | tee -a "$WATCHLOG"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "$WATCHLOG"
pgrep -af 'train_cfg.py' | grep -v grep | tee -a "$WATCHLOG" || echo "train_cfg stopped" | tee -a "$WATCHLOG"
echo "$(date -Is) DONE stop-at-9000 (6000 retained)" | tee -a "$WATCHLOG"
