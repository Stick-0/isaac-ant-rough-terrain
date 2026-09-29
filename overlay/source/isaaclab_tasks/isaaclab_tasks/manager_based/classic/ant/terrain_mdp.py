# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ground-relative height terms for the Ant's generated terrain."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_height_above_ground(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Torso clearance as one observation, preserving the policy's observation size."""
    ground_z = env.scene["ground_height"].data.ray_hits_w[:, 0, 2]
    height = env.scene["robot"].data.root_pos_w[:, 2] - ground_z
    # A ray can miss beyond the finite terrain. Keep observations finite on that
    # terminal step and treat missing ground as a fall in the termination below.
    return torch.nan_to_num(height, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(-1)


def root_height_below_ground_minimum(env: ManagerBasedRLEnv, minimum_height: float) -> torch.Tensor:
    """Terminate below the local ground clearance threshold, rather than world Z."""
    return base_height_above_ground(env).squeeze(-1) < minimum_height
