# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare Ant checkpoints on identical terrain and initial states, one episode per ant.

Example: ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/evaluate_ant.py
    --headless --seed 1001 --checkpoints baseline.pt candidate.pt --output comparison.json

Each environment contributes only its first episode. Automatic resets therefore
cannot inflate the survival rate or mix successive episodes into distance metrics.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoints", nargs="+", required=True, help="Local RSL-RL checkpoint paths.")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=1001)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--episode_details", action="store_true", help="Save paired per-ant metrics and initial-state hashes.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1:
    parser.error("--num_envs must be positive")
for checkpoint in args.checkpoints:
    if not Path(checkpoint).is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import hashlib
import json
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


@torch.inference_mode()
def evaluate(env, runner, checkpoint):
    """Measure deterministic inference, including falls, without sampling policy noise."""
    runner.load(checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    env.seed(args.seed)
    obs, _ = env.reset()
    base = env.unwrapped
    robot = base.scene["robot"]
    initial_state = torch.cat(
        (robot.data.root_state_w, robot.data.joint_pos, robot.data.joint_vel), dim=-1
    )
    initial_state_hash = hashlib.sha256(initial_state.cpu().numpy().tobytes()).hexdigest()
    start = robot.data.root_pos_w[:, :2].clone()
    active = torch.ones(env.num_envs, dtype=torch.bool, device=base.device)
    survived = torch.zeros_like(active)
    failed = torch.zeros_like(active)
    samples = torch.zeros(env.num_envs, device=base.device)
    sway_sum = torch.zeros_like(samples)
    action_delta_sum = torch.zeros_like(samples)
    speed_sum = torch.zeros_like(samples)
    forward_distance = torch.zeros_like(samples)
    previous_actions = torch.zeros(env.num_envs, env.num_actions, device=base.device)

    for step in range(base.max_episode_length):
        if not simulation_app.is_running():
            raise RuntimeError("Simulator closed before evaluation finished")
        actions = policy(obs)
        # Sample before stepping: terminal steps automatically reset the robot.
        # Endpoint displacement therefore excludes at most one simulation step.
        forward_distance[active] = (robot.data.root_pos_w[:, 0] - start[:, 0])[active]
        sway_sum[active] += robot.data.root_ang_vel_b[:, :2].square().sum(dim=-1)[active]
        action_delta_sum[active] += (actions - previous_actions).square().mean(dim=-1)[active]
        speed_sum[active] += robot.data.root_lin_vel_w[:, 0][active]
        samples[active] += 1
        previous_actions.copy_(actions)
        obs, _, dones, _ = env.step(actions)
        ended = active & dones.bool()
        failed[ended] = base.reset_terminated[ended]
        survived[ended] = base.reset_time_outs[ended] & ~base.reset_terminated[ended]
        active[ended] = False
        if not active.any():
            break

    if active.any():
        raise RuntimeError(f"{active.sum().item()} episodes did not finish within the configured limit")
    values = {
        "survival_rate": survived.float().mean().item(),
        "failed_episodes": failed.sum().item(),
        "episode_duration_mean_s": (samples * base.step_dt).mean().item(),
        "observed_forward_distance_mean_m": forward_distance.mean().item(),
        "forward_speed_mean_m_s": (speed_sum / samples).mean().item(),
        "roll_pitch_angular_speed_rms_rad_s": (sway_sum / samples).sqrt().mean().item(),
        "action_delta_rms_per_joint": (action_delta_sum / samples).sqrt().mean().item(),
    }
    if not all(torch.isfinite(torch.tensor(value)) for value in values.values()):
        raise RuntimeError(f"Nonfinite evaluation metrics: {values}")
    result = {"checkpoint": str(Path(checkpoint).resolve()), **values}
    print(f"[EVAL] {json.dumps(result)}", flush=True)
    if args.episode_details:
        result["checkpoint_sha256"] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        result["initial_state_sha256"] = initial_state_hash
        result["episodes"] = {
            "survived": survived.cpu().tolist(),
            "failed": failed.cpu().tolist(),
            "duration_s": (samples * base.step_dt).cpu().tolist(),
            "forward_distance_m": forward_distance.cpu().tolist(),
            "forward_speed_m_s": (speed_sum / samples).cpu().tolist(),
            "body_sway_rms_rad_s": (sway_sum / samples).sqrt().cpu().tolist(),
            "action_delta_rms": (action_delta_sum / samples).sqrt().cpu().tolist(),
        }
    return result


def main():
    task = "Isaac-Ant-v0"
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    # Explicit terrain seed makes held-out geometry independent of RNG consumption.
    env_cfg.scene.terrain.terrain_generator.seed = args.seed
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    env = RslRlVecEnvWrapper(gym.make(task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
        report = {
            "task": task,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "episode_length_s": env_cfg.episode_length_s,
            "distance_note": "Position sampled before terminal step; excludes at most one step.",
            "results": [],
        }
        if args.episode_details:
            report["evaluation_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for checkpoint in args.checkpoints:
            report["results"].append(evaluate(env, runner, checkpoint))
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
