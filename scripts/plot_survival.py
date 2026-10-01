#!/usr/bin/env python3
"""Rebuild the main three-model chart from the common five-seed evaluation.

Requires requirements-analysis.txt. Korean labels use NanumGothic if available;
otherwise labels fall back to English. Bars are pooled means; dots are map means.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

from analyze_models import ROOT, MODELS, load_results, summarize

COLORS = {"flat": "#8898aa", "rough": "#147d92", "recovery": "#168372"}


def main():
    protocol, reports = load_results()
    summary = summarize(protocol, reports)
    # Register installed fonts even when Matplotlib's cached list predates them.
    for path in font_manager.findSystemFonts():
        if Path(path).name in {"NanumGothic.ttf", "NanumGothicBold.ttf"}:
            font_manager.fontManager.addfont(path)
    korean = "NanumGothic" in {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": "NanumGothic" if korean else "DejaVu Sans",
        "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "text.color": "#172b3a", "axes.labelcolor": "#172b3a",
    })
    labels = ({"flat": "평지 모델", "rough": "험지 모델", "recovery": "추가 학습 모델"}
              if korean else {"flat": "Flat-trained", "rough": "Rough-trained", "recovery": "Fine-tuned"})
    fig, ax = plt.subplots(figsize=(10.5, 6.1))
    fig.subplots_adjust(top=.77, bottom=.25, left=.09, right=.97)
    fig.suptitle("같은 새 험지에서 세 모델의 생존율 비교" if korean else "Three models on the same unseen rough terrain",
                 fontsize=21, fontweight="bold", y=.97)
    subtitle = ("동일한 5개 지형 · 동일한 시작 상태 · 모델별 2,560개 첫 에피소드"
                if korean else "Same 5 maps and initial states · 2,560 first episodes per model")
    fig.text(.53, .885, subtitle, ha="center", fontsize=12, color="#526575")
    values = [summary["models"][m]["survival_rate"] * 100 for m in MODELS]
    bars = ax.bar(range(3), values, width=.58, color=[COLORS[m] for m in MODELS])
    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=14,
                 fontsize=19, fontweight="bold")
    for i, model in enumerate(MODELS):
        points = [next(row["survival_rate"] for row in report["results"]
                       if Path(row["checkpoint"]).stem == model) * 100 for report in reports]
        ax.scatter([i + (j - 2) * .065 for j in range(len(points))], points,
                   color="#172b3a", edgecolor="white", linewidth=.65, s=29, zorder=3)
    counts = [summary["models"][m] for m in MODELS]
    ax.set_xticks(range(3), [f"{labels[m]}\n{row['survived_episodes']:,} / {row['episodes']:,}"
                            for m, row in zip(MODELS, counts)], fontsize=13, linespacing=1.7)
    ax.set_ylim(0, 109)
    ax.set_xlim(-.65, 2.65)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.tick_params(length=0, pad=9)
    ax.grid(axis="y", color="#dfe6ec", linewidth=.8)
    ax.set_axisbelow(True)
    footer = ("생존 = 낙상 없이 16초 완료 · 막대 = 전체 평균 · 점 = 지형별 평균\n"
              "지형 seed 5001–5005 · 학습 seed 1개 · 세 모델 모두 지면 상대 높이 관측 사용"
              if korean else
              "Survival = completing 16 seconds without falling · bars = pooled means · dots = map means\n"
              "Terrain seeds 5001–5005 · one training seed · common ground-relative-height observation")
    fig.text(.53, .08, footer, ha="center", va="center", fontsize=10, color="#526575", linespacing=1.9)
    destination = ROOT / "artifacts/figures/survival_comparison.png"
    fig.savefig(destination, dpi=170, facecolor="white")
    plt.close(fig)
    print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
