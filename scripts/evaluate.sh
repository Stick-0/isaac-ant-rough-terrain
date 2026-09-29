#!/usr/bin/env bash
set -euo pipefail

: "${ISAACLAB_ROOT:?Set ISAACLAB_ROOT to the patched IsaacLab_RS directory}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$#" -eq 0 ]; then
    set -- 2001 2002
fi
for terrain_seed in "$@"; do
    TERM=xterm "$ISAACLAB_ROOT/isaaclab.sh" -p \
        "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/evaluate_ant.py" \
        --headless --seed "$terrain_seed" --num_envs 512 \
        --checkpoints "$repo_root/artifacts/models/flat.pt" \
            "$repo_root/artifacts/models/rough.pt" "$repo_root/artifacts/models/stable.pt" \
        --output "$repo_root/results/reproduced/seed_${terrain_seed}.json"
done
