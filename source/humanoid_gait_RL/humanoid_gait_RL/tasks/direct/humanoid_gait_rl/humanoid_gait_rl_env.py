# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform, quat_apply_inverse, yaw_quat, \
                                quat_from_euler_xyz, quat_mul, quat_from_angle_axis, \
                                wrap_to_pi
                                

from .humanoid_gait_rl_env_cfg import HumanoidGaitRlEnvCfg


def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)

class HumanoidGaitRlEnv(DirectRLEnv):
    cfg: HumanoidGaitRlEnvCfg

    def __init__(self, cfg: HumanoidGaitRlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dof_idx, _ = self.robot.find_joints([".*"])
        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)
                
        self.last_action = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.heading_control_stiffness = 0.5 # large value makes the robot follow the command more rapidly
        self.resampling_command_time = 10.0 # every 10 seconds resample commands
        self.action_scale = 0.5
        
    def _resample_commands(self, env_ids: torch.Tensor):           
        num_resamples = len(env_ids)  
                                            
        # resample x and y velocity command
        if self.cfg.is_play:
            self.commands[env_ids, 0] = torch.ones(num_resamples, device=self.device)  # lin_vel_x: [0, 1]                          
        else:
            self.commands[env_ids, 0] = torch.rand(num_resamples, device=self.device)  # lin_vel_x: [0, 1]              
        self.commands[env_ids, 1] = torch.zeros(num_resamples, device=self.device)  # lin_vel_y: [0, 0]          
                
        # resample direction
        self.target_heading[env_ids] = torch.rand(num_resamples, device=self.device) * 2 * math.pi - math.pi  # [-pi, pi]
        current_heading = self.robot.data.heading_w[env_ids] 
        heading_error = wrap_to_pi(self.target_heading[env_ids] - current_heading)
        self.commands[env_ids, 2] = torch.clamp(  
            heading_error * self.heading_control_stiffness,  
            min=-1.0,  
            max=1.0  
        ) # ang_vel_z [-1, 1]
        
        # reset command timer
        self.command_time_left[env_ids] = self.resampling_command_time
                    
    def _setup_scene(self):
        # create assets
        self.robot = Articulation(self.cfg.robot_cfg)
        self.height_scanner = RayCaster(self.cfg.height_scanner)
        self.ankle_contact_sensor = ContactSensor(self.cfg.ankle_contact_forces)
        self.torso_contach_sensor = ContactSensor(self.cfg.torso_contact_forces)
        
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors['height_scanner'] = self.height_scanner
        self.scene.sensors['ankle_contact_sensor'] = self.ankle_contact_sensor
        self.scene.sensors['torso_contact_sensor'] = self.torso_contach_sensor
        
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        # initialize commands
        self.commands = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)  # [vel_x, vel_y, ang_vel_z]
        self.target_heading = torch.zeros(self.cfg.scene.num_envs, device=self.device)          
                
        # initialize command timer
        self.command_time_left = torch.zeros(self.cfg.scene.num_envs, device=self.device)                        
                    
        # visualize markers      
        self.visualization_markers = define_markers()
        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset[:,-1] = 0.5
        self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4), device=self.device)
        self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4), device=self.device)

    def _visualize_markers(self):                
        # get marker locations and orientations
        self.marker_locations = self.robot.data.root_pos_w
        self.forward_marker_orientations = self.robot.data.root_quat_w
        self.command_marker_orientations = quat_from_angle_axis(self.target_heading, self.up_dir).squeeze()

        # offset markers so they are above the jetbot
        loc = self.marker_locations + self.marker_offset
        loc = torch.vstack((loc, loc))
        rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

        # render the markers
        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
        self.visualization_markers.visualize(loc, rots, marker_indices=indices)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()        
        
        # update z angular velocity commands
        current_heading = self.robot.data.heading_w  
        heading_error = wrap_to_pi(self.target_heading - current_heading)
        self.commands[:, 2] = torch.clamp(  
            heading_error * self.heading_control_stiffness,   
            min=-1.0,   
            max=1.0  
        )
        
        # resample commands every 10 seconds
        self.command_time_left -= self.step_dt
        resample_env_ids = (self.command_time_left <= 0).nonzero(as_tuple=False).flatten()  
        if len(resample_env_ids) > 0:  
            self._resample_commands(resample_env_ids)
            
        self._visualize_markers()
    
    def _apply_action(self) -> None:   
        # get the default joint positions and apply the action
        # target_position = default_position + action        
        default_joint_pos = self.robot.data.default_joint_pos[:, self.dof_idx]  
        target_joint_pos = default_joint_pos + self.action_scale * self.actions
        
        self.robot.set_joint_position_target(target_joint_pos, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:                
        base_lin_vel = self.robot.data.root_com_lin_vel_b
        base_ang_vel = self.robot.data.root_com_ang_vel_b
        projected_gravity = self.robot.data.projected_gravity_b
        commands = self.commands        
        joint_pos = self.robot.data.joint_pos - self.robot.data.default_joint_pos
        joint_vel = self.robot.data.joint_vel - self.robot.data.default_joint_vel  
        actions = self.last_action      
                        
        height_scan = self.height_scanner.data.pos_w[:, 2].unsqueeze(1) - self.height_scanner.data.ray_hits_w[..., 2] - 0.5
        height_scan = torch.clip(height_scan, -1.0, 1.0)         
                
        obs = torch.hstack((base_lin_vel, 
                            base_ang_vel, 
                            projected_gravity, 
                            commands,                             
                            joint_pos, 
                            joint_vel, 
                            actions, 
                            height_scan))
        
        observations = {"policy": obs}
        
        return observations

    def _get_rewards(self) -> torch.Tensor:        
        # termination penalty
        termination_penalty = self._get_dones()[0]        
                
        # track_lin_vel_xy_exp        
        vel_yaw = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), self.robot.data.root_lin_vel_w[:, :3])
        track_lin_vel_xy_exp = torch.sum(
            torch.square(self.commands[:, :2] - vel_yaw[:, :2]), dim=1
        )        
        track_lin_vel_xy_exp = torch.exp(-track_lin_vel_xy_exp / 0.5**2)
        
        # track_ang_vel_z_exp
        ang_vel_error = torch.square(self.commands[:, 2] - self.robot.data.root_ang_vel_w[:, 2])
        track_ang_vel_z_exp = torch.exp(-ang_vel_error / 0.5**2)                        
        
        # ang_vel_xy_l2
        ang_vel_xy_l2 = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)
                
        # dof_torques_l2 
        spe_joint_ids, _ = self.robot.find_joints([".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])
        dof_torques_l2 = torch.sum(torch.square(self.robot.data.applied_torque[:, spe_joint_ids]), dim=1)
                
        # dof_acc_l2
        spe_joint_ids, _ = self.robot.find_joints([".*_hip_.*", ".*_knee_joint"])
        dof_acc_l2 = torch.sum(torch.square(self.robot.data.joint_acc[:, spe_joint_ids]), dim=1)
        
        # action_rate_l2
        action_rate_l2 = torch.sum(torch.square(self.actions - self.last_action), dim=1)
        self.last_action = self.actions.clone()
        
        # feet_air_time        
        air_time = self.ankle_contact_sensor.data.current_air_time
        contact_time = self.ankle_contact_sensor.data.current_contact_time
        in_contact = contact_time > 0.0
        in_mode_time = torch.where(in_contact, contact_time, air_time)
        single_stance = torch.sum(in_contact.int(), dim=1) == 1
        feet_air_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
        feet_air_time = torch.clamp(feet_air_time, max=0.4)
        feet_air_time *= torch.norm(self.commands[:, :2], dim=1) > 0.1
                
        # feet_slide
        ankle_sensor_ids = [0, 1]
        contacts = self.ankle_contact_sensor.data.net_forces_w_history[:, :, ankle_sensor_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        body_vel = self.robot.data.body_lin_vel_w[:, ankle_sensor_ids, :2]
        feet_slide = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)    
        
        # dof_pos_limits
        spe_joint_ids, _ = self.robot.find_joints([".*_ankle_pitch_joint", ".*_ankle_roll_joint"])
        out_of_limits = -(self.robot.data.joint_pos[:, spe_joint_ids] - self.robot.data.soft_joint_pos_limits[:, spe_joint_ids, 0]).clip(max=0.0)
        out_of_limits += (self.robot.data.joint_pos[:, spe_joint_ids] - self.robot.data.soft_joint_pos_limits[:, spe_joint_ids, 1]).clip(min=0.0)
        dof_pos_limits = torch.sum(out_of_limits, dim=1)
        
        # joint_deviation_hip
        spe_joint_ids, _ = self.robot.find_joints([".*_hip_yaw_joint", ".*_hip_roll_joint"])
        angle = self.robot.data.joint_pos[:, spe_joint_ids] - self.robot.data.default_joint_pos[:, spe_joint_ids]
        joint_deviation_hip = torch.sum(torch.abs(angle), dim=1)
        
        # joint_deviation_arms
        spe_joint_ids, _ = self.robot.find_joints([
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint"])
        angle = self.robot.data.joint_pos[:, spe_joint_ids] - self.robot.data.default_joint_pos[:, spe_joint_ids]
        joint_deviation_arms = torch.sum(torch.abs(angle), dim=1)
        
        # joint_deviation_fingers
        spe_joint_ids, _ = self.robot.find_joints([
                ".*_five_joint",
                ".*_three_joint",
                ".*_six_joint",
                ".*_four_joint",
                ".*_zero_joint",
                ".*_one_joint",
                ".*_two_joint",])
        angle = self.robot.data.joint_pos[:, spe_joint_ids] - self.robot.data.default_joint_pos[:, spe_joint_ids]
        joint_deviation_fingers = torch.sum(torch.abs(angle), dim=1)
                
        # joint_deviation_torso
        spe_joint_ids, _ = self.robot.find_joints(["torso_joint"])
        angle = self.robot.data.joint_pos[:, spe_joint_ids] - self.robot.data.default_joint_pos[:, spe_joint_ids]
        joint_deviation_torso = torch.sum(torch.abs(angle), dim=1)
        
        # flat_orientation_l2
        flat_orientation_l2 = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)                       
                
        total_reward = \
            -200 * termination_penalty + \
            1.0 * track_lin_vel_xy_exp + \
            2.0 * track_ang_vel_z_exp + \
            -0.05 * ang_vel_xy_l2 + \
            -1.5e-7 * dof_torques_l2 + \
            -1.25e-7 * dof_acc_l2 + \
            -0.005 * action_rate_l2 + \
            -1.0 * dof_pos_limits + \
            -0.1 * joint_deviation_hip + \
            -0.1 * joint_deviation_arms + \
            -0.05 * joint_deviation_fingers + \
            -0.1 * joint_deviation_torso + \
            0.25 * feet_air_time + \
            -0.1 * feet_slide + \
            -1.0 * flat_orientation_l2

        return self.step_dt * total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
                    
        # If torso is in contact with the ground, the episode is over
        net_contact_forces = self.torso_contach_sensor.data.net_forces_w_history
        illegal_contact = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, [0]], dim=-1), dim=1)[0] > 1.0, dim=1)
        
        return illegal_contact, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
                
        self._resample_commands(env_ids) 
        self.last_action = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        
        # reset force and torque
        force_range = (0.0, 0.0)
        torque_range = (-0.0, 0.0)
        spe_link_ids, _ = self.robot.find_bodies(["torso_link"])
        num_bodies = len(spe_link_ids)
        size = (len(env_ids), num_bodies, 3)
        forces = sample_uniform(*force_range, size, self.device)
        torques = sample_uniform(*torque_range, size, self.device)        
        self.robot.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=spe_link_ids)                                                               

        # set the root state for the reset envs
        pose_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}
        velocity_range = {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            }
        root_states = self.robot.data.default_root_state[env_ids].clone()
        
        # poses
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        
        positions = root_states[:, 0:3] + self.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = quat_mul(root_states[:, 3:7], orientations_delta)
        
        # velocities
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        
        velocities = root_states[:, 7:13] + rand_samples
        
        self.robot.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)                             
                
        # reset joint state
        if self.dof_idx != slice(None):
            iter_env_ids = env_ids[:, None]
        else:
            iter_env_ids = env_ids
        
        position_range = (1.0, 1.0)
        velocity_range = (0.0, 0.0)
        
        joint_pos = self.robot.data.default_joint_pos[iter_env_ids, self.dof_idx].clone()
        joint_vel = self.robot.data.default_joint_vel[iter_env_ids, self.dof_idx].clone()

        joint_pos *= sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
        joint_vel *= sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

        joint_pos_limits = self.robot.data.soft_joint_pos_limits[iter_env_ids, self.dof_idx]
        joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

        joint_vel_limits = self.robot.data.soft_joint_vel_limits[iter_env_ids, self.dof_idx]
        joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=self.dof_idx, env_ids=env_ids)
        
        self._visualize_markers()
