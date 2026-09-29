#!/usr/bin/env python3
"""Analyze the matched-budget reward comparison, preserving paired episodes."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'rough': 'Rough baseline', 'stable': 'Old stability +300',
          'control': 'Original reward +600', 'recovery': 'New reward +600'}
METRICS = {'survived': 'survival_rate', 'forward_distance_m': 'observed_forward_distance_mean_m',
           'forward_speed_m_s': 'forward_speed_mean_m_s', 'duration_s': 'episode_duration_mean_s',
           'body_sway_rms_rad_s': 'roll_pitch_angular_speed_rms_rad_s',
           'action_delta_rms': 'action_delta_rms_per_joint'}


def paired_difference(groups, metric, candidate='recovery', baseline='control'):
    """Resample matched ant IDs within each map; conditional on these two maps/policies."""
    differences = [np.asarray(g[candidate]['episodes'][metric], dtype=float)
                   - np.asarray(g[baseline]['episodes'][metric], dtype=float) for g in groups]
    rng = np.random.default_rng(20260929)
    boot = np.zeros(5000)
    total = sum(len(d) for d in differences)
    for delta in differences:
        # Equal ant ID gets both policies; resample separately within each terrain seed.
        boot += delta[rng.integers(0, len(delta), (len(boot), len(delta)))].sum(axis=1) / total
    return {'mean': float(np.concatenate(differences).mean()),
            'paired_bootstrap_95_percent_interval': np.quantile(boot, [.025, .975]).tolist()}


def read_reports(folder):
    reports = [json.loads(p.read_text()) for p in sorted(folder.glob('seed_*.json'))]
    if {r['seed'] for r in reports} != {4001, 4002} or len(reports) != 2:
        raise ValueError('Final comparison requires exactly the predeclared seeds 4001 and 4002')
    groups = []
    for report in reports:
        assert report['num_envs'] == 512 and report['episode_length_s'] == 16
        group = {Path(row['checkpoint']).stem: row for row in report['results']}
        assert len(report['results']) == len(MODELS) and set(group) == set(MODELS)
        assert len({row['initial_state_sha256'] for row in group.values()}) == 1
        for row in group.values():
            for metric, aggregate in METRICS.items():
                values = np.asarray(row['episodes'][metric], dtype=float)
                assert len(values) == report['num_envs'] and np.isfinite(values).all()
                assert np.isclose(values.mean(), row[aggregate], atol=1e-5)
            failed = np.asarray(row['episodes']['failed'], dtype=bool)
            survived = np.asarray(row['episodes']['survived'], dtype=bool)
            assert np.logical_xor(failed, survived).all()
            assert failed.sum() == row['failed_episodes']
        groups.append(group)
    return reports, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results/rewards/test')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/rewards/summary')
    args = parser.parse_args()
    reports, groups = read_reports(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for model in MODELS:
        rows = [group[model] for group in groups]
        summary[model] = {'episodes': 1024, 'failed_episodes': sum(row['failed_episodes'] for row in rows)}
        for key in METRICS.values():
            summary[model][key] = sum(row[key] for row in rows) / len(rows)
    comparisons = {}
    for baseline in ('rough', 'stable', 'control'):
        comparisons[baseline] = {metric: paired_difference(groups, metric, baseline=baseline)
                                 for metric in METRICS}
        rescued = lost = 0
        for group in groups:
            a = np.asarray(group['recovery']['episodes']['survived'], dtype=bool)
            b = np.asarray(group[baseline]['episodes']['survived'], dtype=bool)
            rescued += int((a & ~b).sum())
            lost += int((~a & b).sum())
        comparisons[baseline]['recovery_only_survived'] = rescued
        comparisons[baseline]['baseline_only_survived'] = lost
    payload = {'models': summary, 'recovery_minus_baseline': comparisons,
               'interval_note': 'Paired ant-ID bootstrap within each fixed map, 5000 replicates. '
               'Conditional on these maps and trained policies; not training-seed or map-population uncertainty.'}
    (args.output_dir / 'aggregate.json').write_text(json.dumps(payload, indent=2) + '\n')
    table = ['| Model | Completion | Falls / 1024 | Distance | Speed | Action RMS | Body RMS |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for model, row in summary.items():
        table.append(f"| {MODELS[model]} | {row['survival_rate']*100:.2f}% | {row['failed_episodes']} | "
                     f"{row['observed_forward_distance_mean_m']:.2f} m | {row['forward_speed_mean_m_s']:.2f} m/s | "
                     f"{row['action_delta_rms_per_joint']:.4f} | {row['roll_pitch_angular_speed_rms_rad_s']:.4f} rad/s |")
    (args.output_dir / 'table.md').write_text('\n'.join(table) + '\n')
    with (args.output_dir / 'per_seed.csv').open('w') as stream:
        writer = csv.DictWriter(stream, ['seed', 'model', 'failed_episodes', *METRICS.values()],
                                extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        for report, group in zip(reports, groups):
            for model, row in group.items():
                writer.writerow({'seed': report['seed'], 'model': model, **row})

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout='constrained')
    colors = ['#9aa8b5', '#c5a069', '#3b80a5', '#168372']
    for ax, key, title, factor in zip(axes, ['survival_rate', 'observed_forward_distance_mean_m',
                                           'roll_pitch_angular_speed_rms_rad_s'],
                                     ['16 s completion (%)', 'Forward distance (m)', 'Body angular speed RMS (rad/s)'],
                                     [100, 1, 1]):
        values = [summary[m][key] * factor for m in MODELS]
        bars = ax.bar(range(4), values, color=colors, width=.64)
        ax.bar_label(bars, fmt='%.2f', padding=6, fontsize=10)
        for i, model in enumerate(MODELS):
            ax.scatter([i-.065, i+.065], [g[model][key]*factor for g in groups], color='#162b39', s=18, zorder=3)
        ax.set_xticks(range(4), ['Rough', 'Old\n+300', 'Control\n+600', 'New\n+600'])
        ax.set_title(title, loc='left', fontsize=12)
        ax.set_ylim(0, max(values) * 1.2)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('Reward redesign · matched additional training budget', fontsize=17, fontweight='bold')
    fig.supxlabel('512 first episodes × 2 new maps per policy · dots = map means · one training seed', fontsize=10)
    (ROOT / 'artifacts/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / 'artifacts/figures/reward_comparison.png', dpi=170)
    plt.close(fig)
    print('\n'.join(table))
    print(json.dumps(comparisons['control'], indent=2))


if __name__ == '__main__':
    main()
