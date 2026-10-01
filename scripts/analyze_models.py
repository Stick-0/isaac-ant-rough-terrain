#!/usr/bin/env python3
"""Validate and aggregate the common five-seed evaluation of the three main models."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"flat": "평지", "rough": "험지", "recovery": "추가 학습"}
METRICS = {
    "survived": "survival_rate",
    "duration_s": "episode_duration_mean_s",
    "forward_distance_m": "observed_forward_distance_mean_m",
    "forward_speed_m_s": "forward_speed_mean_m_s",
    "body_sway_rms_rad_s": "roll_pitch_angular_speed_rms_rad_s",
    "action_delta_rms": "action_delta_rms_per_joint",
}


def load_results(results_dir=ROOT / "results/models"):
    protocol = json.loads((ROOT / "results/models/protocol.json").read_text())
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert protocol["models"] == list(MODELS)
    evaluator = ROOT / "overlay/scripts/reinforcement_learning/rsl_rl/evaluate_ant.py"
    assert hashlib.sha256(evaluator.read_bytes()).hexdigest() == protocol["evaluation_script_sha256"]
    files = sorted(results_dir.glob("seed_*.json"))
    assert {p.name for p in files} == {f"seed_{s}.json" for s in protocol["terrain_seeds"]}
    reports = [json.loads(p.read_text()) for p in files]
    for path, report in zip(files, reports):
        assert path.name == f"seed_{report['seed']}.json"
        assert report["task"] == protocol["task"]
        assert report["num_envs"] == protocol["num_envs_per_seed"]
        assert report["episode_length_s"] == protocol["episode_length_s"]
        assert report["evaluation_script_sha256"] == protocol["evaluation_script_sha256"]
        assert [Path(row["checkpoint"]).stem for row in report["results"]] == list(MODELS)
        assert len({row["initial_state_sha256"] for row in report["results"]}) == 1
        for row in report["results"]:
            model = Path(row["checkpoint"]).stem
            assert row["checkpoint_sha256"] == protocol["checkpoint_sha256"][model]
            assert row["checkpoint_sha256"] == manifest["models"][model]["sha256"]
            assert row["checkpoint"] == manifest["models"][model]["path"]
            assert len(row["initial_state_sha256"]) == 64
            for key, metric in METRICS.items():
                values = row["episodes"][key]
                assert len(values) == report["num_envs"]
                assert all(math.isfinite(value) for value in values)
                assert math.isclose(sum(values) / len(values), row[metric], abs_tol=1e-5)
            failed, survived = row["episodes"]["failed"], row["episodes"]["survived"]
            assert len(failed) == report["num_envs"]
            assert all(type(v) is bool for v in failed + survived)
            assert all(f != s for f, s in zip(failed, survived))
            assert sum(failed) == row["failed_episodes"]
            assert all(0 < t <= protocol["episode_length_s"] for t in row["episodes"]["duration_s"])
    assert {report["seed"] for report in reports} == set(protocol["terrain_seeds"])
    return protocol, reports


def summarize(protocol, reports):
    groups = [{Path(row["checkpoint"]).stem: row for row in report["results"]} for report in reports]
    models = {}
    for model in MODELS:
        rows = [group[model] for group in groups]
        count = sum(len(row["episodes"]["survived"]) for row in rows)
        assert count == protocol["first_episodes_per_model"]
        failed = sum(row["failed_episodes"] for row in rows)
        result = {"episodes": count, "survived_episodes": count - failed, "failed_episodes": failed}
        for key, metric in METRICS.items():
            result[metric] = sum(sum(row["episodes"][key]) for row in rows) / count
        models[model] = result
    comparisons = {}
    for baseline, candidate in (("flat", "rough"), ("rough", "recovery")):
        a, b = models[baseline], models[candidate]
        rescued = lost = 0
        for group in groups:
            pairs = zip(group[baseline]["episodes"]["survived"], group[candidate]["episodes"]["survived"])
            for before, after in pairs:
                rescued += int(after and not before)
                lost += int(before and not after)
        comparisons[f"{candidate}_vs_{baseline}"] = {
            "survival_difference_percentage_points": 100 * (b["survival_rate"] - a["survival_rate"]),
            "fall_reduction_percent": 100 * (1 - b["failed_episodes"] / a["failed_episodes"]),
            "speed_change_percent": 100 * (b["forward_speed_mean_m_s"] / a["forward_speed_mean_m_s"] - 1),
            "distance_change_percent": 100 * (b["observed_forward_distance_mean_m"] / a["observed_forward_distance_mean_m"] - 1),
            "body_sway_change_percent": 100 * (b["roll_pitch_angular_speed_rms_rad_s"] / a["roll_pitch_angular_speed_rms_rad_s"] - 1),
            "action_delta_change_percent": 100 * (b["action_delta_rms_per_joint"] / a["action_delta_rms_per_joint"] - 1),
            "candidate_only_survived": rescued,
            "baseline_only_survived": lost,
        }
        assert rescued - lost == a["failed_episodes"] - b["failed_episodes"]
    return {"terrain_seeds": protocol["terrain_seeds"], "models": models, "comparisons": comparisons,
            "note": "All three models share all five maps and matched initial states. First episodes only; one training seed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/models")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/models/summary")
    args = parser.parse_args()
    protocol, reports = load_results(args.results_dir)
    summary = summarize(protocol, reports)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = ["seed", "model", "episodes", "failed_episodes", *METRICS.values()]
    with (args.output_dir / "per_seed.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for report in reports:
            for row in report["results"]:
                writer.writerow({**row, "seed": report["seed"], "model": Path(row["checkpoint"]).stem,
                                 "episodes": report["num_envs"]})
    table = ["| 모델 | 생존율 | 낙상 / 2,560회 | 평균 거리 | 평균 속도 | 몸체 각속도 RMS | 액션 변화 RMS |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model, row in summary["models"].items():
        table.append(f"| {MODELS[model]} | {row['survival_rate'] * 100:.2f}% | {row['failed_episodes']:,} | "
                     f"{row['observed_forward_distance_mean_m']:.2f}m | {row['forward_speed_mean_m_s']:.2f}m/s | "
                     f"{row['roll_pitch_angular_speed_rms_rad_s']:.3f} rad/s | {row['action_delta_rms_per_joint']:.4f} |")
    (args.output_dir / "table.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))
    print(json.dumps(summary["comparisons"], indent=2))


if __name__ == "__main__":
    main()
