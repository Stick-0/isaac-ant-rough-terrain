# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Conservative stability fine-tuning with the existing 60-dimensional observation."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

from .ant_env_cfg import AntEnvCfg, RewardsCfg


@configclass
class StableRewardsCfg(RewardsCfg):
    # Penalize changes in commanded torque, rather than reducing all torque.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # Penalize roll/pitch angular velocity without penalizing yaw.
    body_sway = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.025)


@configclass
class AntStableEnvCfg(AntEnvCfg):
    """Keep terrain, observations, actions and termination rules identical to the baseline."""

    rewards: StableRewardsCfg = StableRewardsCfg()
