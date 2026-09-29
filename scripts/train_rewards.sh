#!/usr/bin/env bash
set -euo pipefail
: "${ISAACLAB_ROOT:?Set ISAACLAB_ROOT to the patched IsaacLab_RS directory}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ISAACLAB_ROOT"
# Equal data budget, initial weights, PPO optimizer, terrain and random seed.
TERM=xterm "$ISAACLAB_ROOT/isaaclab.sh" -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Ant-Stable-v0 --headless --num_envs 4096 --seed 42 \
    --max_iterations 600 --run_name reward_control \
    --warm_start "$repo_root/artifacts/models/rough.pt" \
    agent.experiment_name=ant_reward_control \
    env.rewards.action_rate.weight=0.0 env.rewards.body_sway.weight=0.0
TERM=xterm "$ISAACLAB_ROOT/isaaclab.sh" -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Ant-Recovery-v0 --headless --num_envs 4096 --seed 42 \
    --max_iterations 600 --run_name reward_recovery \
    --warm_start "$repo_root/artifacts/models/rough.pt"
