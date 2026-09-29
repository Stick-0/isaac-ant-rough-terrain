# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Risk-sensitive Ant rewards; all terms except the fall cost are rates per second."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .terrain_mdp import base_height_above_ground

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def capped_target_speed(
    env: ManagerBasedRLEnv, speed_cap: float, target_pos: tuple[float, float, float]
) -> torch.Tensor:
    """Reward forward motion without paying extra for speed above the cap.

    Use planar velocity projected toward the original target. This avoids the
    large-potential subtraction in the original progress term. There is no
    positive progress reward for standing still, and reverse motion is negative.
    """
    robot = env.scene["robot"].data
    direction = robot.root_pos_w.new_tensor(target_pos[:2]) - robot.root_pos_w[:, :2]
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    speed = (robot.root_lin_vel_w[:, :2] * direction).sum(dim=-1)
    return speed.clamp(min=-speed_cap, max=speed_cap)


def low_clearance_risk(env: ManagerBasedRLEnv, safe_height: float, fall_height: float) -> torch.Tensor:
    """Quadratic warning before the unchanged local-ground fall threshold.

    Heights above the safe threshold have no penalty: do not reward jumping or
    force a precise torso height on uneven ground. Missing rays count as risk.
    """
    height = base_height_above_ground(env).squeeze(-1)
    return ((safe_height - height) / (safe_height - fall_height)).clamp(0.0, 1.0).square()


def tilt_risk(env: ManagerBasedRLEnv, safe_up: float, critical_up: float) -> torch.Tensor:
    """Penalize large tilt, with a deadband for normal gait and terrain slopes."""
    up = -env.scene["robot"].data.projected_gravity_b[:, 2]
    return ((safe_up - up) / (safe_up - critical_up)).clamp(0.0, 1.0).square()


def fall_event(env: ManagerBasedRLEnv) -> torch.Tensor:
    """A one-off fall cost after RewardManager's multiplication by step_dt.

    Pure time limits incur no cost. A real fall on the final step still does.
    Termination is computed before rewards, and resets happen afterwards.
    """
    return env.termination_manager.terminated.float() / env.step_dt
