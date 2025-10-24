# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab_assets import G1_MINIMAL_CFG 

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip



@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,        
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )    
    
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    

@configclass
class HumanoidGaitRlEnvCfg(DirectRLEnvCfg):
    # env
    is_play = False
    decimation = 4
    episode_length_s = 20.0
    
    # - spaces definition
    action_space = 37
    observation_space = 310
    state_space = 0
        
    # robot(s)
    robot_cfg: ArticulationCfg = G1_MINIMAL_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # scene
    scene: MySceneCfg = MySceneCfg(
        num_envs=4096, 
        env_spacing=2.5, 
        replicate_physics=True
    )
    
    # simulation
    physx_cfg: PhysxCfg = PhysxCfg(gpu_max_rigid_patch_count = 10 * 2**15)
    sim: SimulationCfg = SimulationCfg(
        dt=0.005, 
        render_interval=decimation,
        physics_material=scene.terrain.physics_material,
        physx=physx_cfg
    )
        
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=decimation * sim.dt
    )
    
    ankle_contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_ankle_roll_link",
        history_length=3, 
        track_air_time=True,
        update_period=sim.dt
    )    
    
    torso_contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link",
        history_length=3, 
        track_air_time=True,
        update_period=sim.dt
    )


@configclass
class HumanoidGaitRlEnvCfg_PLAY(HumanoidGaitRlEnvCfg):
    is_play = True