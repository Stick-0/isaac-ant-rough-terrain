#!/usr/bin/env python3
"""Rebuild the README survival chart from the two separate evaluation sets.

Uses requirements-analysis.txt. Install fonts-nanum for Korean chart labels;
otherwise English labels are used. No simulation or model training is required.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"flat": "#8898aa", "rough": "#147d92", "recovery": "#168372"}
PANELS = [
    ("results/transfer", (2001, 2002), ("flat", "rough")),
    ("results/rewards/test", (4001, 4002), ("rough", "recovery")),
]


def aggregate(folder, seeds, models):
    totals = {model: {"episodes": 0, "survived": 0} for model in models}
    for seed in seeds:
        report = json.loads((ROOT / folder / f"seed_{seed}.json").read_text())
        assert report["seed"] == seed and report["episode_length_s"] == 16
        assert report["num_envs"] == 512
        rows = {Path(row["checkpoint"]).stem: row for row in report["results"]}
        for model in models:
            row = rows[model]
            survived = report["num_envs"] - row["failed_episodes"]
            assert math.isclose(survived / report["num_envs"], row["survival_rate"])
            totals[model]["episodes"] += report["num_envs"]
            totals[model]["survived"] += survived
    return totals


def main():
    korean = "NanumGothic" in {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": "NanumGothic" if korean else "DejaVu Sans",
        "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "text.color": "#172b3a", "axes.labelcolor": "#172b3a",
    })
    labels = ({"flat": "평지 모델", "rough": "험지 모델", "recovery": "추가 학습 모델"}
              if korean else {"flat": "Flat-trained", "rough": "Rough-trained", "recovery": "Fine-tuned"})
    titles = ["험지 적응", "추가 학습의 효과"] if korean else ["Rough-terrain adaptation", "Further training"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    fig.subplots_adjust(top=.70, bottom=.23, left=.07, right=.98, wspace=.23)
    fig.suptitle("새 험지에서의 16초 생존율" if korean else "16-second survival on unseen rough terrain",
                 fontsize=22, fontweight="bold", y=.97)
    for index, (ax, (folder, seeds, models)) in enumerate(zip(axes, PANELS)):
        totals = aggregate(folder, seeds, models)
        values = [100 * totals[m]["survived"] / totals[m]["episodes"] for m in models]
        bars = ax.bar(range(2), values, width=.52, color=[COLORS[m] for m in models])
        ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=7,
                     fontsize=16, fontweight="bold")
        ax.set_title(f"{index + 1}. {titles[index]}  (+{values[1] - values[0]:.2f}%p)\n"
                     f"Terrain seeds: {seeds[0]} / {seeds[1]}", fontsize=13, pad=24, linespacing=1.6)
        ax.set_xticks(range(2), [f"{labels[m]}\n{totals[m]['survived']:,} / {totals[m]['episodes']:,}"
                                for m in models], linespacing=1.7)
        ax.set_ylim(0, 100)
        ax.set_xlim(-.65, 1.65)
        ax.set_yticks(range(0, 101, 20))
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.tick_params(length=0, pad=9)
        ax.grid(axis="y", color="#dfe6ec", linewidth=.8)
        ax.set_axisbelow(True)
        print(f"seeds={seeds}: " + ", ".join(f"{m}={v:.8f}%" for m, v in zip(models, values)))
    footer = ("각 비교는 동일 지형 · 모델별 첫 에피소드 1,024회 · 두 패널의 평가 지형은 서로 다름\n"
              "생존 = 낙상 없이 16초 완료 · 학습 seed 1개"
              if korean else
              "Matched maps within each panel; 1,024 first episodes per model. Panels use different maps.\n"
              "Survival = completing 16 seconds without falling. One training seed.")
    fig.text(.5, .065, footer, ha="center", va="center", fontsize=10, color="#526575", linespacing=1.8)
    destination = ROOT / "artifacts/figures/survival_comparison.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170, facecolor="white")
    plt.close(fig)
    print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
