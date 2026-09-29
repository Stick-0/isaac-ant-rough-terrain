# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Reward-only stability revision: same terrain, 60 observations and fall rules."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

from . import recovery_mdp
from .ant_env_cfg import AntEnvCfg, RewardsCfg


@configclass
class RecoveryRewardsCfg(RewardsCfg):
    progress = RewTerm(
        func=recovery_mdp.capped_target_speed,
        weight=1.0,
        params={"speed_cap": 4.5, "target_pos": (1000.0, 0.0, 0.0)},
    )
    # Keep the original alive, upright, heading, effort and joint-limit terms.
    # A fall costs ten reward units once, independently of the control timestep.
    fall = RewTerm(func=recovery_mdp.fall_event, weight=-10.0)
    clearance_risk = RewTerm(
        func=recovery_mdp.low_clearance_risk,
        weight=-2.0,
        params={"safe_height": 0.48, "fall_height": 0.31},
    )
    tilt_risk = RewTerm(
        func=recovery_mdp.tilt_risk,
        weight=-2.0,
        params={"safe_up": 0.93, "critical_up": 0.5},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    body_sway = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.025)


@configclass
class AntRecoveryEnvCfg(AntEnvCfg):
    rewards: RecoveryRewardsCfg = RecoveryRewardsCfg()
