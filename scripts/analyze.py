#!/usr/bin/env python3
"""Recompute aggregate tables and static figures from published measurements."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = {
    "survival_rate": ("16 s completion (%)", 100),
    "observed_forward_distance_mean_m": ("Observed forward distance (m)", 1),
    "action_delta_rms_per_joint": ("Action change RMS per joint", 1),
    "roll_pitch_angular_speed_rms_rad_s": ("Roll/pitch angular speed RMS (rad/s)", 1),
}
LABELS = {"flat": "Flat-trained", "rough": "Rough-trained", "stable": "Stability fine-tuned"}
COLORS = ["#8b98a8", "#147d92", "#dca443"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/transfer")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/summary")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for file in sorted(args.results_dir.glob("seed_*.json")):
        report = json.loads(file.read_text())
        for result in report["results"]:
            rows.append({"seed": report["seed"], "episodes": report["num_envs"],
                         "model": Path(result["checkpoint"]).stem, **result})
    if not rows:
        raise SystemExit("No evaluation reports found")
    fields = ["seed", "model", "episodes", "failed_episodes", *METRICS,
              "forward_speed_mean_m_s", "episode_duration_mean_s"]
    with (args.output_dir / "per_seed.csv").open("w") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore", lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for model in LABELS:
        samples = [row for row in rows if row["model"] == model]
        total = sum(row["episodes"] for row in samples)
        if not total:
            raise ValueError(f"Missing results for {model}")
        summary[model] = {"episodes": total, "failed_episodes": sum(s["failed_episodes"] for s in samples)}
        for metric in [*METRICS, "forward_speed_mean_m_s", "episode_duration_mean_s"]:
            summary[model][metric] = sum(s[metric] * s["episodes"] for s in samples) / total
    (args.output_dir / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = ["| Model | Episodes | Completion | Forward distance | Action change RMS | Body angular speed RMS |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for model, values in summary.items():
        table.append(f"| {LABELS[model]} | {values['episodes']} | {values['survival_rate'] * 100:.2f}% | "
                     f"{values['observed_forward_distance_mean_m']:.2f} m | "
                     f"{values['action_delta_rms_per_joint']:.4f} | "
                     f"{values['roll_pitch_angular_speed_rms_rad_s']:.4f} rad/s |")
    (args.output_dir / "table.md").write_text("\n".join(table) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    for ax, (metric, (title, factor)) in zip(axes.flat, METRICS.items()):
        values = [summary[model][metric] * factor for model in LABELS]
        bars = ax.bar(range(3), values, color=COLORS, width=0.6)
        ax.bar_label(bars, fmt="%.3f" if "RMS" in title else "%.2f", padding=8)
        for index, model in enumerate(LABELS):
            points = [r[metric] * factor for r in rows if r["model"] == model]
            ax.scatter([index] * len(points), points, color="#172b4d", s=24, zorder=3)
        ax.set_xticks(range(3), ["Flat\ntrained", "Rough\ntrained", "Stability\nfine-tuned"])
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylim(0, max(values + [r[metric] * factor for r in rows]) * 1.25)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    figure.suptitle("Ant transfer to unseen rough terrain\n2 terrain seeds · 512 first episodes per model per seed",
                    fontsize=16, fontweight="bold")
    figure.supxlabel("Dots = terrain-seed means, not confidence intervals. One training seed; common height adapter.",
                     fontsize=9)
    figures = ROOT / "artifacts/figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "transfer_comparison.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained", sharey=True)
    for ax, model, color in zip(axes, LABELS, COLORS):
        with (ROOT / "results/training" / f"{model}.csv").open() as f:
            records = list(csv.DictReader(f))
        key = "Train/mean_episode_length"
        samples = [(int(r["iteration"]), float(r[key])) for r in records if r[key]]
        ax.plot([x for x, _ in samples], [y / 60 for _, y in samples], color=color, linewidth=0.6, alpha=0.5)
        means = [sum(y for _, y in samples[max(0, i-19):i+1]) / len(samples[max(0, i-19):i+1]) / 60
                 for i in range(len(samples))]
        ax.plot([x for x, _ in samples], means, color=color, linewidth=2)
        ax.set_title(LABELS[model])
        ax.set_xlabel("Iteration within this run")
        ax.set_ylim(0, 17)
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Training mean episode length (s)")
    figure.suptitle("Training history · flat and rough use different training terrain", fontweight="bold")
    figure.savefig(figures / "training_history.png", dpi=170)
    plt.close(figure)
    print("\n".join(table))


if __name__ == "__main__":
    main()
