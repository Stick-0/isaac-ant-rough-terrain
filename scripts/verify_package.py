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
        print(f"PASS: {len(index['assets'])} media hashes, capture timelines and identical rough-terrain resets")
    assert "<!-- RESULTS_TABLE -->" not in (ROOT / "README.md").read_text()
    print(f"PASS: {len(manifest['models'])} checkpoint hashes, 3072 transfer episodes, aggregate values, "
          f"{len(python_files)} Python files and public paths")


if __name__ == "__main__":
    main()
