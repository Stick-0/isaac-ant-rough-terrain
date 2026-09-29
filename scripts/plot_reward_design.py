#!/usr/bin/env python3
"""Plot the configured shaping terms (reward rates, not measured policy results)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
v = np.linspace(-2, 8, 300)
axes[0].plot(v, v, '--', color='#9aa8b5', label='Uncapped progress rate')
axes[0].plot(v, np.clip(v, -4.5, 4.5), color='#168372', label='New capped speed rate')
axes[0].set(xlabel='Speed toward target (m/s)', ylabel='Reward / second', title='No extra reward above 4.5 m/s')
axes[0].legend(fontsize=8)
h = np.linspace(.25, .65, 300)
axes[1].plot(h, -2*np.clip((.48-h)/(.48-.31), 0, 1)**2, color='#168372')
axes[1].axvline(.31, linestyle='--', color='#bd5a4e')
axes[1].text(.32, -1.8, 'Fall: -10 once', color='#bd5a4e', fontsize=10)
axes[1].set(xlabel='Torso clearance above local ground (m)', ylabel='Reward / second', title='Warn before the unchanged fall threshold')
tilt = np.linspace(0, 90, 300)
axes[2].plot(tilt, -2*np.clip((.93-np.cos(np.radians(tilt)))/(.93-.5), 0, 1)**2, color='#168372')
axes[2].set(xlabel='Torso tilt from vertical (degrees)', ylabel='Reward / second', title='Deadband allows normal gait and slopes')
for ax in axes:
    ax.grid(alpha=.15)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Reward design · mathematical curves, not experimental results', fontsize=15, fontweight='bold')
fig.savefig(ROOT/'artifacts/figures/reward_design.png', dpi=170)
plt.close(fig)
