"""LIBERO-10 Task 0 eval for the CFG step-30000 deploy weights.

Two scored runs, same rollout settings as the earlier vanilla evals
(seed 7, 520 env steps, action chunk 10, 10 denoise steps):

  * vanilla: task-text prompt, 50 trials. Comparable to the 6000/9000/12000 numbers.
  * cfg_positive: prompt suffix "Advantage: positive". With cfgrl_guidance_scale=1.0
    the official mixer is exactly this conditional branch, so one forward is the
    standard CFG inference, not a second unconditioned pass.

The CFG run then keeps sampling until 10 successful videos are saved, or 200
trials, whichever comes first. Trials after the first 50 are not in the rate.
"""

import json
import logging
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

from toolkits.standalone_eval_scripts.openpi import setup_policy

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
RES = 256
SCORED_TRIALS = 50
MAX_TRIALS_CFG = 200
MIN_SUCCESS_VIDEOS = 10
MAX_STEPS = 520
ACTION_CHUNK = 10
NUM_STEPS = 10
SEED = 7
TASK_ID = 0

ROOT = pathlib.Path("/root/rlinf_recap/logs/libero_eval_cfg30000_task0_x50")
PRETRAINED = "/root/rlinf_recap/deploy/cfg_rl_step30000"

MODES = (
    {
        "name": "vanilla",
        "prompt_suffix": "",
        "extend_for_videos": False,
        "max_trials": SCORED_TRIALS,
        "note": (
            "vanilla pi05 forward; same rollout as cfg6000/9000/12000; "
            "Task 0 x 50"
        ),
    },
    {
        "name": "cfg_positive",
        "prompt_suffix": "\nAdvantage: positive",
        "extend_for_videos": True,
        "max_trials": MAX_TRIALS_CFG,
        "note": (
            "CFG positive branch; cfgrl_guidance_scale=1.0 so v=v_cond only; "
            "prompt is task text plus 'Advantage: positive'; "
            "first 50 trials are the scored rate; later trials only collect "
            f"success videos up to {MIN_SUCCESS_VIDEOS}"
        ),
    },
)


def quat2axisangle(quat):
    quat = np.array(quat, dtype=np.float64).copy()
    quat[3] = min(1.0, max(-1.0, quat[3]))
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def setup_logger():
    ROOT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(ROOT / "eval.log", mode="a"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("cfg30000_task0")


def load_progress(path):
    if not path.exists():
        return {
            "scored_successes": [],
            "extra_trials": 0,
            "success_videos": [],
        }
    data = json.loads(path.read_text())
    data.setdefault("scored_successes", [])
    data.setdefault("extra_trials", 0)
    data.setdefault("success_videos", [])
    return data


def save_progress(path, progress, **extra):
    payload = {
        "scored_successes": progress["scored_successes"],
        "extra_trials": progress["extra_trials"],
        "success_videos": progress["success_videos"],
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2))


def run_episode(env, policy, init_state, prompt, record_frames):
    env.reset()
    obs = env.set_init_state(init_state)
    policy.reset()
    frames = []
    t = 0
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
        action_chunk = policy.infer(
            {
                "observation/image": img,
                "observation/wrist_image": wrist,
                "observation/state": state,
                "prompt": prompt,
            }
        )["actions"]
        for action in action_chunk[:ACTION_CHUNK]:
            obs, reward, done, info = env.step(action.tolist())
            if record_frames:
                frames.append(
                    np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                )
            t += 1
            success = bool(done and reward)
            if done or t >= MAX_STEPS + 10:
                break
        if done or t >= MAX_STEPS + 10:
            break
    return success, t, frames


def run_mode(policy, mode, task_description, init_states, env, logger):
    log_dir = ROOT / mode["name"]
    video_dir = log_dir / "videos"
    log_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "summary.json"
    progress_path = log_dir / "progress.json"
    if summary_path.exists():
        logger.info("SKIP %s; summary already exists", mode["name"])
        return json.loads(summary_path.read_text())

    progress = load_progress(progress_path)
    prompt = f"{task_description}{mode['prompt_suffix']}"
    logger.info(
        "MODE %s prompt=%r scored=%d max_trials=%d extend=%s resume_scored=%d extra=%d videos=%d",
        mode["name"],
        prompt,
        SCORED_TRIALS,
        mode["max_trials"],
        mode["extend_for_videos"],
        len(progress["scored_successes"]),
        progress["extra_trials"],
        len(progress["success_videos"]),
    )

    t0 = time.time()
    next_trial = len(progress["scored_successes"]) + progress["extra_trials"]
    pbar = tqdm.tqdm(total=mode["max_trials"], initial=next_trial, desc=mode["name"])
    while next_trial < mode["max_trials"]:
        scored_done = len(progress["scored_successes"]) >= SCORED_TRIALS
        videos_done = len(progress["success_videos"]) >= MIN_SUCCESS_VIDEOS
        if scored_done and (videos_done or not mode["extend_for_videos"]):
            break

        is_scored = not scored_done
        record_frames = len(progress["success_videos"]) < MIN_SUCCESS_VIDEOS
        success, steps, frames = run_episode(
            env,
            policy,
            init_states[next_trial % len(init_states)],
            prompt,
            record_frames,
        )
        if is_scored:
            progress["scored_successes"].append(float(success))
        else:
            progress["extra_trials"] += 1

        video_path = None
        if success and record_frames and frames:
            idx = len(progress["success_videos"])
            video_path = video_dir / f"success_{idx:02d}_trial{next_trial:03d}.mp4"
            imageio.mimwrite(video_path, frames, fps=20)
            progress["success_videos"].append(
                {
                    "index": idx,
                    "trial": next_trial,
                    "scored": is_scored,
                    "steps": steps,
                    "path": str(video_path),
                }
            )
            logger.info("saved success video %s", video_path)

        scored = progress["scored_successes"]
        logger.info(
            "%s trial %d success=%s steps=%d scored=%d/%d rate=%.3f videos=%d",
            mode["name"],
            next_trial,
            success,
            steps,
            len(scored),
            SCORED_TRIALS,
            float(np.mean(scored)) if scored else 0.0,
            len(progress["success_videos"]),
        )
        save_progress(
            progress_path,
            progress,
            mode=mode["name"],
            next_trial=next_trial + 1,
            running_success_rate=float(np.mean(scored)) if scored else 0.0,
        )
        next_trial += 1
        pbar.update(1)
    pbar.close()

    scored = [float(x) for x in progress["scored_successes"][:SCORED_TRIALS]]
    summary = {
        "checkpoint": PRETRAINED,
        "task_suite": "libero_10",
        "task_id": TASK_ID,
        "task": task_description,
        "mode": mode["name"],
        "prompt": prompt,
        "num_scored_trials": len(scored),
        "successes": scored,
        "success_rate": float(np.mean(scored)) if scored else 0.0,
        "extra_trials_for_videos": progress["extra_trials"],
        "success_videos_saved": len(progress["success_videos"]),
        "success_videos_target": MIN_SUCCESS_VIDEOS,
        "success_videos": progress["success_videos"],
        "max_trials_cap": mode["max_trials"],
        "action_chunk": ACTION_CHUNK,
        "num_steps": NUM_STEPS,
        "max_env_steps": MAX_STEPS,
        "seed": SEED,
        "note": mode["note"],
        "render": "osmesa",
        "elapsed_sec": time.time() - t0,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "DONE %s success_rate=%.3f videos=%d extra_trials=%d",
        mode["name"],
        summary["success_rate"],
        summary["success_videos_saved"],
        summary["extra_trials_for_videos"],
    )
    return summary


def main():
    logger = setup_logger()
    logger.info("Task0 x50 eval start checkpoint=%s", PRETRAINED)
    t0 = time.time()
    policy = setup_policy(
        SimpleNamespace(
            config_name="pi05_libero",
            pretrained_path=PRETRAINED,
            num_steps=NUM_STEPS,
        )
    )
    logger.info("Policy loaded in %.1fs", time.time() - t0)

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(TASK_ID)
    init_states = suite.get_task_init_states(TASK_ID)
    task_description = task.language
    bddl = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES
    )
    env.seed(SEED + TASK_ID)
    logger.info("Task %d: %s init_states=%d", TASK_ID, task_description, len(init_states))

    summaries = []
    try:
        for mode in MODES:
            summaries.append(
                run_mode(policy, mode, task_description, init_states, env, logger)
            )
    finally:
        env.close()

    combined = {
        "checkpoint": PRETRAINED,
        "task_id": TASK_ID,
        "task": task_description,
        "modes": summaries,
        "elapsed_sec": time.time() - t0,
    }
    (ROOT / "summary.json").write_text(json.dumps(combined, indent=2))
    logger.info("ALL_DONE %s", ROOT / "summary.json")
    print("SUMMARY", json.dumps(combined))


if __name__ == "__main__":
    main()
