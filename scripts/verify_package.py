#!/usr/bin/env python3
"""Verify published checkpoints, measurements, aggregate values and Python syntax."""

import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, model in manifest["models"].items():
        data = (ROOT / model["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == model["sha256"], f"Hash mismatch: {name}"
    reports = list((ROOT / "results/transfer").glob("seed_*.json"))
    assert {json.loads(p.read_text())["seed"] for p in reports} == {2001, 2002}
    all_rows = []
    for report_path in [*reports, *(ROOT / "results/stability").glob("*.json")]:
        report = json.loads(report_path.read_text())
        assert report["num_envs"] == 512
        assert report["episode_length_s"] == 16
        for result in report["results"]:
            assert (ROOT / result["checkpoint"]).is_file(), result["checkpoint"]
            assert math.isclose(result["survival_rate"] * 512 + result["failed_episodes"], 512)
            assert 0 < result["episode_duration_mean_s"] <= 16
            for value in result.values():
                if isinstance(value, (int, float)):
                    assert math.isfinite(value)
        if report_path in reports:
            assert {Path(r["checkpoint"]).stem for r in report["results"]} == {"flat", "rough", "stable"}
            all_rows.extend(report["results"])
    aggregate = json.loads((ROOT / "results/summary/aggregate.json").read_text())
    for name, values in aggregate.items():
        rows = [r for r in all_rows if Path(r["checkpoint"]).stem == name]
        assert len(rows) == 2
        assert values["episodes"] == 1024
        assert values["failed_episodes"] == sum(r["failed_episodes"] for r in rows)
        for metric, value in values.items():
            if metric not in {"episodes", "failed_episodes"}:
                assert math.isclose(value, sum(r[metric] for r in rows) / 2, rel_tol=1e-12)
    reward_reports = list((ROOT / "results/rewards/test").glob("seed_*.json"))
    if reward_reports:
        assert {json.loads(p.read_text())["seed"] for p in reward_reports} == {4001, 4002}
        mappings = {
            "survived": "survival_rate", "duration_s": "episode_duration_mean_s",
            "forward_distance_m": "observed_forward_distance_mean_m",
            "forward_speed_m_s": "forward_speed_mean_m_s",
            "body_sway_rms_rad_s": "roll_pitch_angular_speed_rms_rad_s",
            "action_delta_rms": "action_delta_rms_per_joint",
        }
        collected = {name: [] for name in ("rough", "stable", "control", "recovery")}
        for report_path in [*reward_reports, *(ROOT / "results/rewards/development").glob("seed_*.json")]:
            report = json.loads(report_path.read_text())
            assert report["num_envs"] == 512 and report["episode_length_s"] == 16
            assert len(report["results"]) == 4
            assert {Path(row["checkpoint"]).stem for row in report["results"]} == set(collected)
            assert len({row["initial_state_sha256"] for row in report["results"]}) == 1
            for row in report["results"]:
                assert (ROOT / row["checkpoint"]).is_file()
                model = Path(row["checkpoint"]).stem
                assert row["checkpoint_sha256"] == manifest["models"][model]["sha256"]
                assert len(row["initial_state_sha256"]) == 64
                for key, metric in mappings.items():
                    values = row["episodes"][key]
                    assert len(values) == 512 and all(math.isfinite(v) for v in values)
                    assert math.isclose(sum(values) / 512, row[metric], abs_tol=1e-5)
                assert len(row["episodes"]["failed"]) == 512
                assert sum(row["episodes"]["failed"]) == row["failed_episodes"]
                assert all(a != b for a, b in zip(row["episodes"]["survived"], row["episodes"]["failed"]))
                assert all(0 < v <= 16 for v in row["episodes"]["duration_s"])
                if report_path in reward_reports:
                    collected[Path(row["checkpoint"]).stem].append(row)
        summary = json.loads((ROOT / "results/rewards/summary/aggregate.json").read_text())
        for model, rows in collected.items():
            assert len(rows) == 2
            target = summary["models"][model]
            assert target["episodes"] == 1024
            assert target["failed_episodes"] == sum(row["failed_episodes"] for row in rows)
            for metric in mappings.values():
                assert math.isclose(target[metric], sum(row[metric] for row in rows) / 2, abs_tol=1e-8)
        for baseline, comparison in summary["recovery_minus_baseline"].items():
            expected = (summary["models"]["recovery"]["survival_rate"]
                        - summary["models"][baseline]["survival_rate"])
            assert math.isclose(comparison["survived"]["mean"], expected, abs_tol=1e-12)
            count_delta = comparison["recovery_only_survived"] - comparison["baseline_only_survived"]
            assert math.isclose(count_delta / 1024, expected, abs_tol=1e-12)
        print("PASS: 4096 reward-test episodes, paired resets and episode-level aggregate consistency")
    from analyze_models import load_results, summarize

    protocol, common_reports = load_results()
    common_summary = summarize(protocol, common_reports)
    saved_summary = json.loads((ROOT / "results/models/summary/aggregate.json").read_text())
    assert saved_summary == common_summary, "Common-evaluation aggregate is stale"
    primary = manifest["primary_evaluation"]
    assert primary["terrain_seeds"] == protocol["terrain_seeds"]
    assert primary["models"] == protocol["models"]
    assert primary["first_episodes_per_model"] == protocol["first_episodes_per_model"]
    for relative, expected in primary["report_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    assert len(primary["report_sha256"]) == len(protocol["terrain_seeds"])
    assert len({r["results"][0]["initial_state_sha256"] for r in common_reports}) == len(common_reports)
    print("PASS: 7680 common-evaluation episodes across all three models, five matched maps, "
          "paired initial states, report hashes and aggregates")
    python_files = list((ROOT / "scripts").glob("*.py")) + list((ROOT / "overlay").rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(), filename=str(path))
    for folder in ("configs", "results"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file():
                assert "/home/" not in path.read_text(), f"Local absolute path: {path}"
    media_root = ROOT / "artifacts/media"
    if (media_root / "index.json").exists():
        index = json.loads((media_root / "index.json").read_text())
        for asset in index["assets"]:
            data = (media_root / asset["file"]).read_bytes()
            assert len(data) == asset["bytes"], asset["file"]
            assert hashlib.sha256(data).hexdigest() == asset["sha256"], asset["file"]
        clips = {}
        for name in ("flat_on_flat", "flat_on_rough", "rough_on_rough"):
            clip = json.loads((media_root / (name + ".json")).read_text())
            assert clip["frames"] == 480 and clip["fps"] == 30
            assert clip["simulation_seconds"] == 16 and clip["playback_speed"] == 1
            assert len(clip["telemetry"]) == clip["frames"]
            for i, sample in enumerate(clip["telemetry"]):
                assert math.isclose(sample["time_s"], i / 30, abs_tol=1e-9)
            model = Path(clip["checkpoint"]).stem
            assert clip["checkpoint_sha256"] == manifest["models"][model]["sha256"]
            clips[name] = clip
        for key in ("seed", "num_envs", "initial_state_sha256", "camera_eye_offset", "camera_target_offset"):
            assert clips["flat_on_rough"][key] == clips["rough_on_rough"][key]
        revision = media_root / "reward_revision/recovery_on_rough.json"
        if revision.exists():
            clip = json.loads(revision.read_text())
            assert clip["checkpoint_sha256"] == manifest["models"]["recovery"]["sha256"]
            assert clip["frames"] == 480 and clip["fps"] == 30
            assert clip["simulation_seconds"] == 16 and clip["playback_speed"] == 1
            assert len(clip["telemetry"]) == clip["frames"]
            for i, sample in enumerate(clip["telemetry"]):
                assert math.isclose(sample["time_s"], i / 30, abs_tol=1e-9)
            for key in ("seed", "num_envs", "initial_state_sha256", "camera_eye_offset", "camera_target_offset"):
                assert clip[key] == clips["rough_on_rough"][key]
        print(f"PASS: {len(index['assets'])} media hashes, capture timelines and identical rough-terrain resets")
    assert "<!-- RESULTS_TABLE -->" not in (ROOT / "README.md").read_text()
    print(f"PASS: {len(manifest['models'])} checkpoint hashes, 3072 transfer episodes, aggregate values, "
          f"{len(python_files)} Python files and public paths")


if __name__ == "__main__":
    main()
