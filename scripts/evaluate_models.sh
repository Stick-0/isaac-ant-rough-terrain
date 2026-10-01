#!/usr/bin/env bash
# Evaluate the three published models on the same terrain and initial states.
set -euo pipefail
: "${ISAACLAB_ROOT:?Set ISAACLAB_ROOT to the patched IsaacLab_RS directory}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${ANT_EVAL_OUTPUT_DIR:-$repo_root/results/reproduced_models}"
mkdir -p "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"
if [ "$#" -eq 0 ]; then
    set -- 5001 5002 5003 5004 5005
fi
cd "$ISAACLAB_ROOT"
for terrain_seed in "$@"; do
    report="$output_dir/seed_${terrain_seed}.json"
    TERM=xterm "$ISAACLAB_ROOT/isaaclab.sh" -p \
        "$ISAACLAB_ROOT/scripts/reinforcement_learning/rsl_rl/evaluate_ant.py" \
        --headless --seed "$terrain_seed" --num_envs 512 --episode_details \
        --checkpoints "$repo_root/artifacts/models/flat.pt" \
            "$repo_root/artifacts/models/rough.pt" \
            "$repo_root/artifacts/models/recovery.pt" \
        --output "$report"
    # Keep measurements intact; normalize paths for publication and check pairing.
    python3 - "$report" "$repo_root" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
path, root = map(Path, sys.argv[1:])
report = json.loads(path.read_text())
assert [Path(row['checkpoint']).stem for row in report['results']] == ['flat', 'rough', 'recovery']
assert len({row['initial_state_sha256'] for row in report['results']}) == 1
for row in report['results']:
    row['checkpoint'] = Path(row['checkpoint']).relative_to(root).as_posix()
report['evaluation_date_utc'] = datetime.now(timezone.utc).date().isoformat()
report['observation_note'] = 'All three policies use the same 60D ground-relative-height adapter on rough terrain.'
path.write_text(json.dumps(report, indent=2) + '\n')
print(f"[PAIRED] seed={report['seed']} models=3 initial_state_hash_equal=True", flush=True)
PY
done
