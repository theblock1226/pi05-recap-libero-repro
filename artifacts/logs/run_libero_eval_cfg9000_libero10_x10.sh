#!/usr/bin/env bash
# Libero-10 full suite eval: 10 tasks x 10 trials for CFG step9000 deploy weights.
# Vanilla pi05 forward (no CFG guidance_scale) — longer smoke / baseline numbers.
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

LOGDIR=/root/rlinf_recap/logs/libero_eval_cfg9000_libero10_x10
mkdir -p "$LOGDIR/videos"
cd "$REPO_PATH"

exec /root/rlinf_recap/RLinf/.venv/bin/python -u <<'PY'
import json
import math
import os
import pathlib
import time
from types import SimpleNamespace

import imageio
import numpy as np
import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from toolkits.standalone_eval_scripts.openpi import setup_logger, setup_policy

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
RES = 256
NUM_TRIALS = 10
MAX_STEPS = 520
ACTION_CHUNK = 10
NUM_STEPS = 10
# Save at most 1 video per task to limit disk
NUM_SAVE_VIDEOS_PER_TASK = 1
SEED = 7

LOG_DIR = pathlib.Path("/root/rlinf_recap/logs/libero_eval_cfg9000_libero10_x10")
EXP = "cfg9000_libero10_x10"
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


logger = setup_logger(EXP, str(LOG_DIR))
np.random.seed(SEED)

args = SimpleNamespace(
    config_name="pi05_libero", pretrained_path=PRETRAINED, num_steps=NUM_STEPS
)
t0 = time.time()
logger.info("Loading policy from %s", PRETRAINED)
policy = setup_policy(args)
logger.info("Policy loaded in %.1fs", time.time() - t0)

suite = benchmark.get_benchmark_dict()["libero_10"]()
num_tasks = suite.n_tasks
assert num_tasks == 10, num_tasks

per_task = []
all_success = []
video_dir = LOG_DIR / "videos"
video_dir.mkdir(parents=True, exist_ok=True)

for task_id in range(num_tasks):
    task = suite.get_task(task_id)
    init_states = suite.get_task_init_states(task_id)
    task_description = task.language
    bddl = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    logger.info("=== Task[%d/%d]: %s ===", task_id, num_tasks - 1, task_description)

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES
    )
    env.seed(SEED + task_id)

    task_successes = []
    for trial in tqdm.tqdm(range(NUM_TRIALS), desc=f"task{task_id}"):
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
                if trial < NUM_SAVE_VIDEOS_PER_TASK:
                    frames.append(
                        np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    )
                t += 1
                success = bool(done and reward)
                if done or t >= MAX_STEPS + 10:
                    break
            if done or t >= MAX_STEPS + 10:
                break

        task_successes.append(float(success))
        all_success.append(float(success))
        logger.info(
            "task %d trial %d success=%s steps=%d", task_id, trial, success, t
        )
        if trial < NUM_SAVE_VIDEOS_PER_TASK and frames:
            out_mp4 = video_dir / f"task{task_id:02d}_trial{trial}.mp4"
            imageio.mimwrite(out_mp4, frames, fps=20)
            logger.info("saved video %s (%d frames)", out_mp4, len(frames))

        # checkpoint progress to disk each episode
        partial = {
            "checkpoint": PRETRAINED,
            "task_suite": "libero_10",
            "num_tasks": num_tasks,
            "num_trials_per_task": NUM_TRIALS,
            "completed_episodes": len(all_success),
            "running_success_rate": float(np.mean(all_success)),
            "per_task_so_far": per_task
            + [
                {
                    "task_id": task_id,
                    "task": task_description,
                    "successes": task_successes,
                    "success_rate": float(np.mean(task_successes)),
                }
            ],
        }
        (LOG_DIR / "progress.json").write_text(json.dumps(partial, indent=2))

    rate = float(np.mean(task_successes))
    per_task.append(
        {
            "task_id": task_id,
            "task": task_description,
            "successes": task_successes,
            "success_rate": rate,
        }
    )
    logger.info("task %d done success_rate=%.3f", task_id, rate)
    env.close()

overall = float(np.mean(all_success)) if all_success else 0.0
summary = {
    "checkpoint": PRETRAINED,
    "task_suite": "libero_10",
    "num_tasks": num_tasks,
    "num_trials_per_task": NUM_TRIALS,
    "total_episodes": len(all_success),
    "overall_success_rate": overall,
    "per_task": per_task,
    "note": "vanilla pi05 forward; CFG guidance_scale not applied",
    "render": "osmesa",
    "elapsed_sec": time.time() - t0,
}
(LOG_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
logger.info("DONE overall_success_rate=%.3f", overall)
print("SUMMARY", json.dumps(summary))
PY
