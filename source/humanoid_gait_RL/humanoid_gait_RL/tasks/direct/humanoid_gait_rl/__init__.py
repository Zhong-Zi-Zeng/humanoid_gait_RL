# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="MiiSLab-Humanoid-Gait-Rl-Direct-v0",
    entry_point=f"{__name__}.humanoid_gait_rl_env:HumanoidGaitRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_gait_rl_env_cfg:HumanoidGaitRlEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="MiiSLab-Humanoid-Gait-Rl-Direct-Play-v0",
    entry_point=f"{__name__}.humanoid_gait_rl_env:HumanoidGaitRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_gait_rl_env_cfg:HumanoidGaitRlEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)