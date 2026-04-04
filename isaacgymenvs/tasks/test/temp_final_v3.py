# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from typing import Tuple

import numpy as np
import torch
from isaacgym import gymapi, gymtorch, gymutil

from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import *


class DiabloGraspCustom3(VecTask):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = cfg
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.dof_noise = self.cfg["env"]["dofNoise"]

        # 定義關節名稱
        self.right_arm_names = ["r_sho_pitch", "r_sho_roll", "r_el", "r_wrist"]
        self.left_arm_names = ["l_sho_pitch", "l_sho_roll", "l_el", "l_wrist"]
        self.right_gripper_names = [
            "r_index_base", "r_index_middle", "r_index_tip",
            "r_mid_base", "r_mid_middle", "r_mid_tip",
            "r_thumb_base", "r_thumb_middle", "r_thumb_tip"
        ]
        self.dof_indices = {}
        self.target_height = torch.zeros(self.cfg["env"]["numEnvs"], dtype=torch.float32, device=sim_device)
        self.target_pitch = torch.zeros(self.cfg["env"]["numEnvs"], dtype=torch.float32, device=sim_device)
        self.locked_leg_pos = torch.zeros((self.cfg["env"]["numEnvs"], 4), dtype=torch.float32, device=sim_device)

        self.cfg["env"]["numActions"] = 14 
        self.cfg["env"]["numObservations"] = 92

        self.actions = torch.zeros(
            (self.cfg["env"]["numEnvs"], self.cfg["env"]["numActions"]),
            dtype=torch.float32,
            device=sim_device,
        )
        self.prev_actions = torch.zeros_like(self.actions)
        
        # Rewards
        self.dist_reward_scale = self.cfg["env"]["rewards"].get("distRewardScale", 1.0)
        self.rot_reward_scale = self.cfg["env"]["rewards"].get("rotRewardScale", 1.0)
        self.lift_reward_scale = self.cfg["env"]["rewards"].get("liftRewardScale", 1.0)
        self.grasp_reward_scale = self.cfg["env"]["rewards"].get("graspRewardScale", 1.0)
        self.orientation_reward_scale = self.cfg["env"]["rewards"].get("orientationRewardScale", 1.0)
        self.action_penalty_scale = self.cfg["env"]["rewards"].get("actionPenaltyScale", 1.0)
        self.stability_penalty_scale = self.cfg["env"]["rewards"].get("stabilityPenaltyScale", 1.0)

        self.up_axis = "z"
        self.up_axis_idx = 2
        self.states = {}

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        # 原本攝影機視角
        if not self.headless:
            cam_pos = gymapi.Vec3(0.1, -1.5, 0.8)
            cam_target = gymapi.Vec3(-0.05, 0.0, 0.23)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self._acquire_tensors()
        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[..., 1]
        
        self.gripper_upper_limits = self.dof_upper_limits[self.dof_indices["right_gripper"]]
        self.gripper_lower_limits = self.dof_lower_limits[self.dof_indices["right_gripper"]]
        
        self.controlled_dof_indices = self.dof_indices["right_arm"] + self.dof_indices["right_gripper"]
        all_dofs = set(range(self.num_dofs))
        self.non_controlled_dof_indices = list(all_dofs - set(self.controlled_dof_indices))

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._refresh_tensors()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # Robot Asset
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        diablo_asset_file = "urdf/diab/Diablo_ger/Part_gripper_col_rev/URDF/diablo_erc1.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.use_mesh_materials = True
        asset_options.armature = 0.01 
        diablo_asset = self.gym.load_asset(self.sim, asset_root, diablo_asset_file, asset_options)
        
        self.num_dofs = self.gym.get_asset_dof_count(diablo_asset)
        dof_dict = self.gym.get_asset_dof_dict(diablo_asset)
        self.dof_indices["right_arm"] = [dof_dict[name] for name in self.right_arm_names]
        self.dof_indices["left_arm"] = [dof_dict[name] for name in self.left_arm_names]
        self.dof_indices["right_gripper"] = [dof_dict[name] for name in self.right_gripper_names]
        self.dof_indices["legs"] = [dof_dict[n] for n in ["left_fake_hip_joint", "left_fake_knee_joint", "right_fake_hip_joint", "right_fake_knee_joint"]]

        dof_props = self.gym.get_asset_dof_properties(diablo_asset)
        
        # --- 同步 Custom2 鎖死參數 ---
        damping_map = {
            "head_pan": 100.0, "head_tilt": 100.0,
            "l_sho_pitch": 100.0, "l_sho_roll": 100.0, "l_el": 100.0, "l_wrist": 100.0,
            "r_sho_pitch": 0.3, "r_sho_roll": 0.4, "r_el": 0.1, "r_wrist": 0.00182,
            "left_fake_hip_joint": 500.0, "left_fake_knee_joint": 500.0, "left_wheel_joint": 500.0,
            "right_fake_hip_joint": 500.0, "right_fake_knee_joint": 500.0, "right_wheel_joint": 500.0,
        }
        stiffness_map = {
            "head_pan": 1000000.0, "head_tilt": 1000000.0,
            "l_sho_pitch": 1000000.0, "l_sho_roll": 1000000.0, "l_el": 1000000.0, "l_wrist": 1000000.0,
            "r_sho_pitch": 15, "r_sho_roll": 20, "r_el": 13.82181, "r_wrist": 4.54741,
            "left_fake_hip_joint": 500.0, "left_fake_knee_joint": 500.0, "left_wheel_joint": 1000000.0,
            "right_fake_hip_joint": 500.0, "right_fake_knee_joint": 500.0, "right_wheel_joint": 1000000.0,
        }

        self.dof_lower_limits = []
        self.dof_upper_limits = []

        for i in range(self.num_dofs):
            dof_name = self.gym.get_asset_dof_name(diablo_asset, i)
            dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            if dof_name in damping_map:
                dof_props["stiffness"][i] = stiffness_map[dof_name]
                dof_props["damping"][i] = damping_map[dof_name]
            elif dof_name in self.right_gripper_names:
                dof_props["stiffness"][i] = 20.0
                dof_props["damping"][i] = 0.5
                dof_props["effort"][i] = 0.2
            else:
                dof_props["stiffness"][i] = self.cfg["env"]["stiffness"]
                dof_props["damping"][i] = self.cfg["env"]["damping"]
            
            self.dof_lower_limits.append(dof_props["lower"][i])
            self.dof_upper_limits.append(dof_props["upper"][i])

        self.dof_lower_limits = to_torch(self.dof_lower_limits, device=self.device)
        self.dof_upper_limits = to_torch(self.dof_upper_limits, device=self.device)

        # Table Asset
        table_stand_height = 0.05
        t_opts = gymapi.AssetOptions()
        t_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, 0.3, 0.5, table_stand_height, t_opts)

        # Drill Asset
        drill_root = "/home/erc/isaacgym/python/IsaacGymEnvs/isaacgymenvs/tasks/test"
        drill_file = "drill_centered.urdf"
        drill_opts = gymapi.AssetOptions()
        drill_opts.vhacd_enabled = True
        self.object_asset = self.gym.load_asset(self.sim, drill_root, drill_file, drill_opts)

        self.envs = []
        self.actor_handles = []
        self.object_handles = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            
            # Robot
            diablo_actor = self.gym.create_actor(env_ptr, diablo_asset, gymapi.Transform(p=gymapi.Vec3(0,0,0.25)), "diablo", i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, diablo_actor, dof_props)
            self.actor_handles.append(diablo_actor)

            # Table
            table_pose = gymapi.Transform(p=gymapi.Vec3(0.25, 0.0, 0.35))
            self.gym.create_actor(env_ptr, table_stand_asset, table_pose, "table", i, 1, 0)

            # Drill - 精確定位
            drill_pose = gymapi.Transform(p=gymapi.Vec3(0.20, -0.15, 0.45), r=gymapi.Quat(0,0,1,0))
            object_actor = self.gym.create_actor(env_ptr, self.object_asset, drill_pose, "drill", i, 2, 0)
            self.object_handles.append(object_actor)

            # Markers
            m_opts = gymapi.AssetOptions()
            m_opts.fix_base_link = True
            m_opts.disable_gravity = True
            self.gym.create_actor(env_ptr, self.gym.create_sphere(self.sim, 0.01, m_opts), gymapi.Transform(p=gymapi.Vec3(0.25, -0.1, 0.85)), "target", i, 1, 1)
            zone_actor = self.gym.create_actor(env_ptr, self.gym.create_box(self.sim, 0.2, 0.15, 0.001, m_opts), gymapi.Transform(p=gymapi.Vec3(0.2, -0.15, 0.376)), "zone", i, 3, 1)
            self.gym.set_rigid_body_color(env_ptr, zone_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 1.0))

            self.envs.append(env_ptr)

        self.diablo_actor_ids = torch.arange(0, self.num_envs * 5, 5, dtype=torch.int32, device=self.device)
        self.object_actor_ids = torch.arange(2, self.num_envs * 5, 5, dtype=torch.int32, device=self.device)
        self.eef_handle = self.gym.find_actor_rigid_body_index(self.envs[0], self.actor_handles[0], "panda_grip_site", gymapi.DOMAIN_ENV)
        self.handle_target_handle = self.gym.find_actor_rigid_body_index(self.envs[0], self.object_handles[0], "handle_target", gymapi.DOMAIN_ENV)
        
        self.init_data()

    def init_data(self):
        self._acquire_tensors()
        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float32, device=self.device)
        arm_r_default = np.radians([0, -60, -80, 0])
        for idx, val in zip(self.dof_indices["right_arm"], arm_r_default):
            self.default_dof_pos[idx] = val
        arm_l_default = np.radians([0, 0, 0, 0])
        for idx, val in zip(self.dof_indices["left_arm"], arm_l_default):
            self.default_dof_pos[idx] = val
        self.default_dof_pos[9] = np.radians(-30)
        self.default_dof_pos[10] = np.radians(-80)
        self.default_dof_pos[self.dof_indices["right_gripper"]] = self.dof_lower_limits[self.dof_indices["right_gripper"]]

    def _acquire_tensors(self):
        self.root_state = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.dof_state = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        self.rigid_body_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self.rigid_body_pos = self.rigid_body_state[:, :3].view(self.num_envs, -1, 3)
        self.rigid_body_rot = self.rigid_body_state[:, 3:7].view(self.num_envs, -1, 4)

    def _update_states(self):
        self.states.update({
            "eef_pos": self.rigid_body_pos[:, self.eef_handle],
            "eef_rot": self.rigid_body_rot[:, self.eef_handle],
            "drill_pos": self.root_state[self.object_actor_ids, :3],
            "drill_rot": self.root_state[self.object_actor_ids, 3:7],
            "handle_target_pos": self.rigid_body_pos[:, self.handle_target_handle],
            "handle_target_rot": self.rigid_body_rot[:, self.handle_target_handle],
            "dof_pos": self.dof_state.view(self.num_envs, -1, 2)[..., 0],
            "dof_vel": self.dof_state.view(self.num_envs, -1, 2)[..., 1],
            "default_dof_pos": self.default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1)
        })

    def _refresh_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._update_states()

    def compute_reward(self):
        dummy_finger = torch.zeros_like(self.states["eef_pos"])
        target_marker_pos = torch.tensor([0.25, -0.10, 0.85], device=self.device).repeat(self.num_envs, 1)
        self.rew_buf[:], self.reset_buf[:] = compute_diablo_reward(
            self.reset_buf, self.progress_buf, self.actions,
            self.states["eef_pos"], self.states["handle_target_pos"],
            self.states["eef_rot"], self.states["handle_target_rot"],
            dummy_finger, dummy_finger, dummy_finger,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, 1.0, 1.0,
            1.0, self.action_penalty_scale, self.lift_reward_scale, self.orientation_reward_scale,
            self.max_episode_length,
            self.states["drill_pos"], self.states["drill_rot"], 0.375, 
            target_marker_pos,
            self.root_state[self.diablo_actor_ids, 3:7]
        )

    def compute_observations(self):
        self._refresh_tensors()
        self.obs_buf.fill_(0.0)
        self.obs_buf[:, 0] = self.progress_buf / self.max_episode_length
        self.obs_buf[:, 1:1+self.num_dofs] = self.states["dof_pos"]
        return self.obs_buf

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        R, l1, l2 = 0.08, 0.14, 0.14
        max_leg_length = l1 + l2
        h_target = 0.23 + torch.rand(num_resets, device=self.device) * 0.12
        p_target = (torch.rand(num_resets, device=self.device) - 0.5) * 0.4
        y_target = (torch.rand(num_resets, device=self.device) - 0.5) * 0.4
        L0 = torch.clamp((h_target - R) / torch.cos(p_target), 0.01, max_leg_length)
        alpha = torch.acos(torch.clamp((l1**2 + l2**2 - L0**2) / (2 * l1 * l2), -1.0, 1.0))
        target_knee = 3.14159 - alpha
        target_hip = -p_target - torch.acos(torch.clamp(L0 / (2 * l1), -1.0, 1.0))
        self.locked_leg_pos[env_ids, 0] = target_hip
        self.locked_leg_pos[env_ids, 1] = target_knee
        self.locked_leg_pos[env_ids, 2] = target_hip
        self.locked_leg_pos[env_ids, 3] = target_knee
        leg_indices = self.dof_indices["legs"]
        self.dof_pos[env_ids[:, None], leg_indices] = self.locked_leg_pos[env_ids]
        arm_gripper_indices = self.dof_indices["right_arm"] + self.dof_indices["right_gripper"]
        self.dof_pos[env_ids[:, None], arm_gripper_indices] = self.default_dof_pos[arm_gripper_indices]
        left_arm_indices = self.dof_indices["left_arm"]
        self.dof_pos[env_ids[:, None], left_arm_indices] = self.default_dof_pos[left_arm_indices]
        self.dof_vel[env_ids] = 0.0
        base_x_offset = L0 * torch.sin(p_target) * torch.cos(y_target)
        base_y_offset = L0 * torch.sin(p_target) * torch.sin(y_target)
        self.root_state[self.diablo_actor_ids[env_ids], 0:3] = torch.stack([base_x_offset, base_y_offset, h_target], dim=1)
        self.root_state[self.diablo_actor_ids[env_ids], 3:7] = quat_from_euler_xyz(torch.zeros_like(p_target), p_target, y_target)
        indices = self.object_actor_ids[env_ids].to(torch.int32)
        self.root_state[indices, 0] = 0.20 + (torch.rand(num_resets, device=self.device)-0.5)*0.05
        self.root_state[indices, 1] = -0.15 + (torch.rand(num_resets, device=self.device)-0.5)*0.05
        self.root_state[indices, 2] = 0.45 
        self.root_state[indices, 3:7] = torch.tensor([0,0,1,0], dtype=torch.float32, device=self.device)
        all_ids = torch.cat([self.diablo_actor_ids[env_ids], indices])
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state), gymtorch.unwrap_tensor(all_ids), len(all_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(self.diablo_actor_ids[env_ids]), len(env_ids))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        targets[:] = self.default_dof_pos 
        targets[:, self.dof_indices["legs"]] = self.locked_leg_pos
        targets[:, self.dof_indices["right_arm"]] = self.action_scale * self.actions[:, :4] + self.default_dof_pos[self.dof_indices["right_arm"]]
        eef_pos = self.states["eef_pos"]
        handle_pos = self.states["handle_target_pos"]
        align_dist = torch.linalg.norm(eef_pos - handle_pos, ord=2, dim=-1)
        u_gripper = self.actions[:, 13]
        should_close = (align_dist <= 0.025) & (u_gripper >= 0.0)
        targets[:, self.dof_indices["right_gripper"]] = torch.where(should_close.unsqueeze(1), self.gripper_upper_limits, self.gripper_lower_limits)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(tensor_clamp(targets, self.dof_lower_limits, self.dof_upper_limits)))

    def post_physics_step(self):
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0: self.reset_idx(env_ids)
        self.compute_observations()
        self.compute_reward()

# @torch.jit.script
def compute_diablo_reward(
    reset_buf, progress_buf, actions,
    eef_pos, handle_pos, eef_rot, handle_rot,
    f1, f2, f3, num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, lift_reward_scale, orientation_reward_scale,
    max_episode_length, object_pos, object_rot, initial_object_z, target_marker_pos, robot_rot
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float, float, Tensor, Tensor, float, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    d = torch.norm(eef_pos - handle_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward = torch.where(d <= 0.05, dist_reward * 2.0, dist_reward)
    gripper_forward_axis = torch.tensor([0.0, 0.0, -1.0], device=eef_pos.device).repeat(num_envs, 1)
    gripper_up_axis = torch.tensor([1.0, 0.0, 0.0], device=eef_pos.device).repeat(num_envs, 1)
    handle_inward_axis = torch.tensor([1.0, 0.0, 0.0], device=eef_pos.device).repeat(num_envs, 1)
    handle_up_axis = torch.tensor([0.0, 0.0, 1.0], device=eef_pos.device).repeat(num_envs, 1)
    axis1 = quat_apply(eef_rot, gripper_forward_axis)
    axis2 = quat_apply(handle_rot, handle_inward_axis)
    axis3 = quat_apply(eef_rot, gripper_up_axis)
    axis4 = quat_apply(handle_rot, handle_up_axis)
    dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)
    is_grasping = (d < 0.025) & (actions[:, 13] >= 0.0)
    object_height = object_pos[:, 2] - initial_object_z
    lift_reward = torch.where(is_grasping, 15.0 * torch.clamp(object_height, min=0.0), torch.zeros_like(d))
    dist_to_target = torch.norm(object_pos - target_marker_pos, p=2, dim=-1)
    proximity_reward = torch.where(is_grasping, 5.0 / (1.0 + dist_to_target * 5.0), torch.zeros_like(d))
    is_success = (dist_to_target < 0.05)
    success_reward = torch.where(is_success, torch.ones_like(d) * 20.0, torch.zeros_like(d))
    rewards = dist_reward_scale * dist_reward + rot_reward_scale * rot_reward + lift_reward + proximity_reward + success_reward - 0.1 * torch.sum(actions**2, dim=-1)
    reset_buf = torch.where(is_success | (object_pos[:, 2] < 0.1) | (progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)
    return rewards, reset_buf
