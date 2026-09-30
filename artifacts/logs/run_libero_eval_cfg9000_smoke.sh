#!/usr/bin/env bash
# Minimal Libero smoke eval for CFG step9000 deploy weights.
# Note: standalone OpenPI loader = vanilla pi05 forward (no CFG guidance scale).
set -euo pipefail
export REPO_PATH=/root/rlinf_recap/RLinf
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export HF_HOME=/root/.cache/huggingface

LOGDIR=/root/rlinf_recap/logs/libero_eval_cfg9000_smoke
mkdir -p "$LOGDIR"
cd "$REPO_PATH"

exec /root/rlinf_recap/RLinf/.venv/bin/python -u <<'PY'
import collections
import math
import os
import pathlib
import json
import time

import imageio
import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from types import SimpleNamespace

from toolkits.standalone_eval_scripts.openpi import setup_logger, setup_policy

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
RES = 256
TASK_ID = 0  # RECAP task0
NUM_TRIALS = 3
MAX_STEPS = 520
ACTION_CHUNK = 10
NUM_STEPS = 10
NUM_SAVE_VIDEOS = 2
SEED = 7

LOG_DIR = "/root/rlinf_recap/logs/libero_eval_cfg9000_smoke"
EXP = "cfg9000_libero10_task0_smoke"
PRETRAINED = "/root/rlinf_recap/deploy/cfg_rl_step9000"


def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


logger = setup_logger(EXP, LOG_DIR)
np.random.seed(SEED)

args = SimpleNamespace(config_name="pi05_libero", pretrained_path=PRETRAINED, num_steps=NUM_STEPS)
t0 = time.time()
logger.info("Loading policy from %s", PRETRAINED)
policy = setup_policy(args)
logger.info("Policy loaded in %.1fs", time.time() - t0)

suite = benchmark.get_benchmark_dict()["libero_10"]()
task = suite.get_task(TASK_ID)
init_states = suite.get_task_init_states(TASK_ID)
task_description = task.language
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
logger.info("Task[%s]: %s", TASK_ID, task_description)

env = OffScreenRenderEnv(
    bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES
)
env.seed(SEED)

successes = []
video_dir = pathlib.Path(LOG_DIR) / "videos"
video_dir.mkdir(parents=True, exist_ok=True)

for trial in tqdm.tqdm(range(NUM_TRIALS), desc="trials"):
    env.reset()
    obs = env.set_init_state(init_states[trial % len(init_states)])
    policy.reset()
    t = 0
    frames = []
    done = False
    success = False
    while t < MAX_STEPS + 10:
        if t < 10:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        state = np.concatenate(
            (
                np.array(obs["robot0_eef_pos"]),
                quat2axisangle(np.array(obs["robot0_eef_quat"])),
                np.array(obs["robot0_gripper_qpos"]),
            )
        )
        element = {
            "observation/image": img,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": str(task_description),
        }
        action_chunk = policy.infer(element)["actions"]
        for action in action_chunk[:ACTION_CHUNK]:
            obs, reward, done, info = env.step(action.tolist())
            if trial < NUM_SAVE_VIDEOS:
                frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            t += 1
            success = bool(done and reward)
            if done or t >= MAX_STEPS + 10:
                break
        if done or t >= MAX_STEPS + 10:
            break
    successes.append(float(success))
    logger.info("trial %d success=%s steps=%d", trial, success, t)
    if trial < NUM_SAVE_VIDEOS and frames:
        out_mp4 = video_dir / f"task{TASK_ID}_trial{trial}.mp4"
        imageio.mimwrite(out_mp4, frames, fps=20)
        logger.info("saved video %s (%d frames)", out_mp4, len(frames))

env.close()
rate = float(np.mean(successes)) if successes else 0.0
summary = {
    "checkpoint": PRETRAINED,
    "task_suite": "libero_10",
    "task_id": TASK_ID,
    "task": task_description,
    "num_trials": NUM_TRIALS,
    "successes": successes,
    "success_rate": rate,
    "note": "vanilla pi05 forward; CFG guidance_scale not applied in this smoke loader",
    "render": "osmesa",
}
(pathlib.Path(LOG_DIR) / "summary.json").write_text(json.dumps(summary, indent=2))
logger.info("DONE success_rate=%.3f summary=%s", rate, summary)
print("SUMMARY", json.dumps(summary))
PY
