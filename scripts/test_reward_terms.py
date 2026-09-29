#!/usr/bin/env python3
"""CPU checks of reward semantics without starting Isaac Sim (requires torch)."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'overlay/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant'
# A tiny isolated package imports the real functions, not copied equations.
package = ModuleType('_ant_reward_test')
package.__path__ = [str(SOURCE)]
sys.modules[package.__name__] = package
for name in ('terrain_mdp', 'recovery_mdp'):
    spec = importlib.util.spec_from_file_location(f'{package.__name__}.{name}', SOURCE / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
rewards = module


def make_env(heights, velocities=None, up=None, ground=None, terminated=None, dt=1 / 60):
    n = len(heights)
    heights = torch.tensor(heights, dtype=torch.float)
    root = torch.zeros(n, 3)
    root[:, 2] = heights
    velocity = torch.zeros(n, 3)
    velocity[:, 0] = torch.tensor(velocities or [0.] * n)
    gravity = torch.zeros(n, 3)
    gravity[:, 2] = -torch.tensor(up or [1.] * n)
    hits = torch.zeros(n, 1, 3)
    hits[:, 0, 2] = torch.tensor(ground or [0.] * n)
    return SimpleNamespace(
        scene={
            'robot': SimpleNamespace(data=SimpleNamespace(
                root_pos_w=root, root_lin_vel_w=velocity, projected_gravity_b=gravity)),
            'ground_height': SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits)),
        },
        termination_manager=SimpleNamespace(
            terminated=torch.tensor(terminated or [False] * n),
            time_outs=torch.tensor([True] * n)),
        step_dt=dt,
    )


class RewardTests(unittest.TestCase):
    def test_progress_stopped_reverse_and_overspeed(self):
        env = make_env([.5] * 5, velocities=[0., -2., 3., 4.5, 9.])
        actual = rewards.capped_target_speed(env, 4.5, (1000., 0., 0.))
        torch.testing.assert_close(actual, torch.tensor([0., -2., 3., 4.5, 4.5]))

    def test_direction_uses_planar_target_and_handles_target_reached(self):
        env = make_env([.5, .5], velocities=[3., 3.])
        env.scene['robot'].data.root_pos_w[0, :2] = torch.tensor([1000., 0.])
        env.scene['robot'].data.root_pos_w[1, :2] = torch.tensor([1001., 0.])
        actual = rewards.capped_target_speed(env, 4.5, (1000., 0., 100.))
        torch.testing.assert_close(actual, torch.tensor([0., -3.]))

    def test_clearance_respects_local_ground_and_deadband(self):
        env = make_env([.7, .48, .395, .31, 10.395], ground=[0., 0., 0., 0., 10.])
        actual = rewards.low_clearance_risk(env, .48, .31)
        torch.testing.assert_close(actual, torch.tensor([0., 0., .25, 1., .25]), atol=1e-5, rtol=1e-5)

    def test_missing_ground_is_finite_and_maximum_risk(self):
        env = make_env([.5, .5], ground=[float('nan'), float('inf')])
        actual = rewards.low_clearance_risk(env, .48, .31)
        torch.testing.assert_close(actual, torch.ones(2))

    def test_tilt_deadband_and_inversion(self):
        env = make_env([.5] * 5, up=[1., .93, .715, .5, -1.])
        actual = rewards.tilt_risk(env, .93, .5)
        torch.testing.assert_close(actual, torch.tensor([0., 0., .25, 1., 1.]))

    def test_fall_cost_timestep_independent_and_timeout_not_a_fall(self):
        for dt in (1 / 30, 1 / 60, 1 / 120):
            env = make_env([.5, .2], terminated=[False, True], dt=dt)
            # Both are timeouts; only the second also has a true fall.
            actual = rewards.fall_event(env) * -10. * dt
            torch.testing.assert_close(actual, torch.tensor([0., -10.]))


if __name__ == '__main__':
    unittest.main()
