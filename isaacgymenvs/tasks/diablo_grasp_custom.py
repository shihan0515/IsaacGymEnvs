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


class DiabloGraspCustom(VecTask):
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

        # 定義關節名稱，以便從 URDF 動態獲取索引，避免硬編碼錯誤
        self.right_arm_names = ["r_sho_pitch", "r_sho_roll", "r_el", "r_wrist"]
        self.right_gripper_names = [
            "r_index_base", "r_index_middle", "r_index_tip",
            "r_mid_base", "r_mid_middle", "r_mid_tip",
            "r_thumb_base", "r_thumb_middle", "r_thumb_tip"
        ]
        self.dof_indices = {}
        # 初始化目標高度與俯仰角張量
        self.target_height = torch.zeros(self.cfg["env"]["numEnvs"], dtype=torch.float32, device=sim_device)
        self.target_pitch = torch.zeros(self.cfg["env"]["numEnvs"], dtype=torch.float32, device=sim_device)

        # observation and action space
        # Actions: 4 right arm joints + 9 unused + 1 right gripper control = 14
        self.cfg["env"]["numActions"] = 14 
        self.cfg["env"]["numObservations"] = 92 # Updated to include height and pitch

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
        self.debug_mode = True 

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

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
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs(
            self.num_envs,
            self.cfg["env"]["envSpacing"],
            int(np.sqrt(self.num_envs)),
        )

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        diablo_asset_file = "urdf/diab/Diablo_ger/Part_gripper_col_rev/URDF/diablo_erc1.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True 
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False 
        asset_options.armature = 0.01
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        print(f"Loading asset {diablo_asset_file} from {asset_root}")
        diablo_asset = self.gym.load_asset(self.sim, asset_root, diablo_asset_file, asset_options)
        self.num_diablo_bodies = self.gym.get_asset_rigid_body_count(diablo_asset)
        self.num_diablo_shapes = self.gym.get_asset_rigid_shape_count(diablo_asset)

        self.num_dofs = self.gym.get_asset_dof_count(diablo_asset)
        print("Total DOFs:", self.num_dofs)
        
        dof_dict = self.gym.get_asset_dof_dict(diablo_asset)
        self.dof_indices["right_arm"] = [dof_dict[name] for name in self.right_arm_names]
        self.dof_indices["right_gripper"] = [dof_dict[name] for name in self.right_gripper_names]
        
        # 獲取腿部關節索引
        self.dof_indices["legs"] = [
            dof_dict["left_fake_hip_joint"], dof_dict["left_fake_knee_joint"],
            dof_dict["right_fake_hip_joint"], dof_dict["right_fake_knee_joint"]
        ]
        
        print(f"Dynamic DOF Indices - Right Arm: {self.dof_indices['right_arm']}")
        print(f"Dynamic DOF Indices - Right Gripper: {self.dof_indices['right_gripper']}")
        print(f"Dynamic DOF Indices - Legs: {self.dof_indices['legs']}")

        # self.cfg["env"]["numObservations"] = 1 + self.num_dofs + self.num_dofs + 3 + 4 + 14 + 2
        print(f"Updated numObservations to: {self.cfg['env']['numObservations']}")

        dof_props = self.gym.get_asset_dof_properties(diablo_asset)
        self.dof_lower_limits = []
        self.dof_upper_limits = []

        # --- 解鎖腿部關節，並鎖死其他無用關節 ---
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

        table_stand_height = 0.05
        table_stand_dims = gymapi.Vec3(0.3, 0.5, 0.01)
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, table_stand_dims.x, table_stand_dims.y, table_stand_dims.z, table_asset_options)
        self.num_table_bodies = self.gym.get_asset_rigid_body_count(table_stand_asset)
        self.num_table_shapes = self.gym.get_asset_rigid_shape_count(table_stand_asset)

        mug_asset_file = "/home/erc/isaacgym/assets/urdf/ycb/025_mug/025_mug.urdf"
        mug_asset_options = gymapi.AssetOptions()
        mug_asset_options.fix_base_link = False
        mug_asset_options.use_mesh_materials = True
        mug_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        mug_asset_options.override_com = True
        mug_asset_options.override_inertia = True
        mug_asset_options.vhacd_enabled = True
        mug_asset_options.vhacd_params = gymapi.VhacdParams()
        mug_asset_options.vhacd_params.resolution = 1000
        self.object_asset = self.gym.load_asset(self.sim, "", mug_asset_file, mug_asset_options)
        self.num_object_bodies = self.gym.get_asset_rigid_body_count(self.object_asset)
        self.num_object_shapes = self.gym.get_asset_rigid_shape_count(self.object_asset)
        self._mug_height = 0.1 

        total_bodies = self.num_diablo_bodies + self.num_table_bodies + self.num_object_bodies
        total_shapes = self.num_diablo_shapes + self.num_table_shapes + self.num_object_shapes

        diablo_start_pose = gymapi.Transform()
        diablo_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.25) 
        diablo_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_stand_pos = gymapi.Vec3(0.25, 0.0, 0.45) # 從 0.25 調高到 0.45
        table_stand_pose = gymapi.Transform()
        table_stand_pose.p = table_stand_pos
        table_stand_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.table_surface_pos = table_stand_pos + gymapi.Vec3(0, 0, table_stand_height / 2)

        object_start_pose = gymapi.Transform()
        object_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.initial_object_z = self.table_surface_pos.z + self._mug_height / 2 + 0.002

        self.envs = []
        self.actor_handles = []
        self.object_handles = []
        self.table_handles = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, total_bodies + 1, total_shapes + 1, True)

            diablo_actor = self.gym.create_actor(env_ptr, diablo_asset, diablo_start_pose, "diablo", i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, diablo_actor, dof_props)
            self.actor_handles.append(diablo_actor)

            table_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_pose, "table", i, 1, 0)
            self.table_handles.append(table_actor)

            mug_start_pos_z = self.table_surface_pos.z + self._mug_height / 2 + 0.002
            object_start_pose.p = gymapi.Vec3(self.table_surface_pos.x, self.table_surface_pos.y + np.random.uniform(-0.25, 0.0), mug_start_pos_z)
            object_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0) 
            object_actor = self.gym.create_actor(env_ptr, self.object_asset, object_start_pose, "mug", i, 2, 0) 
            self.object_handles.append(object_actor)

            # --- 增加虛擬參考點 (Target Marker) ---
            target_marker_opts = gymapi.AssetOptions()
            target_marker_opts.fix_base_link = True
            target_marker_opts.disable_gravity = True
            target_marker_asset = self.gym.create_sphere(self.sim, 0.01, target_marker_opts)
            
            # 設定位置：杯子 Y 軸範圍中心，高度為初始高度 + 5cm
            self.target_pos_local = gymapi.Vec3(
                self.table_surface_pos.x,
                self.table_surface_pos.y - 0.10, # 杯子 Y 軸重置範圍的中心
                self.initial_object_z + 0.05    # 初始高度 + 5公分
            )
            target_pose = gymapi.Transform()
            target_pose.p = self.target_pos_local
            
            # 使用碰撞組 -1 且遮罩為 1 來避免與任何物體碰撞
            target_actor = self.gym.create_actor(env_ptr, target_marker_asset, target_pose, "target_marker", i, 1, 1)
            # --- 增加馬克杯重製範圍標記 (Reset Zone Marker) ---
            zone_marker_opts = gymapi.AssetOptions()
            zone_marker_opts.fix_base_link = True
            zone_marker_opts.disable_gravity = True
            # X 範圍約 0.04m, Y 範圍約 0.10m
            zone_marker_asset = self.gym.create_box(self.sim, 0.16, 0.16, 0.001, zone_marker_opts)
            
            # 初始位置預設在桌面上
            zone_pose = gymapi.Transform()
            zone_actor = self.gym.create_actor(env_ptr, zone_marker_asset, zone_pose, 'zone_marker', i, 3, 1)
            self.gym.set_rigid_body_color(env_ptr, zone_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 1.0)) # 藍色

            self.gym.set_rigid_body_color(env_ptr, target_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 1.0, 0.0)) # 綠色

            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)

        self.diablo_actor_ids = torch.arange(0, self.num_envs * 5, 5, dtype=torch.int32, device=self.device)
        self.object_actor_ids = torch.arange(2, self.num_envs * 5, 5, dtype=torch.int32, device=self.device)
        self.zone_actor_ids = torch.arange(4, self.num_envs * 5, 5, dtype=torch.int32, device=self.device)

        self.eef_handle = self.gym.find_actor_rigid_body_index(self.envs[0], self.actor_handles[0], "panda_grip_site", gymapi.DOMAIN_ENV)
        self.handle_target_handle = self.gym.find_actor_rigid_body_index(self.envs[0], self.object_handles[0], "handle_target", gymapi.DOMAIN_ENV)
        
        print(f"EEF Local Index: {self.eef_handle}")
        print(f"Handle Target Local Index: {self.handle_target_handle}")

        self.init_data()

    def init_data(self):
        self._acquire_tensors()

        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float32, device=self.device)
        arm_default = np.radians([0, -30, -80, 0])
        for idx, val in zip(self.dof_indices["right_arm"], arm_default):
            self.default_dof_pos[idx] = val
            
        self.default_dof_pos[9] = np.radians(-30)
        self.default_dof_pos[10] = np.radians(-80)
            
        self.default_dof_pos[self.dof_indices["right_gripper"]] = self.dof_lower_limits[self.dof_indices["right_gripper"]]

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]

        self.rigid_body_state = gymtorch.wrap_tensor(self.rigid_body_state_tensor)
        self.rigid_body_pos = self.rigid_body_state[:, :3].view(self.num_envs, -1, 3)
        self.rigid_body_rot = self.rigid_body_state[:, 3:7].view(self.num_envs, -1, 4)

    def _acquire_tensors(self):
        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.root_state = gymtorch.wrap_tensor(self.root_state_tensor)
        self.dof_state = gymtorch.wrap_tensor(self.dof_state_tensor)
        self.rigid_body_state = gymtorch.wrap_tensor(self.rigid_body_state_tensor)

    def _update_states(self):
        self.states.update(
            {
                "eef_pos": self.rigid_body_pos[:, self.eef_handle],
                "eef_rot": self.rigid_body_rot[:, self.eef_handle],
                "mug_pos": self.root_state[self.object_actor_ids, :3],
                "mug_rot": self.root_state[self.object_actor_ids, 3:7],
                "handle_target_pos": self.rigid_body_pos[:, self.handle_target_handle],
                "handle_target_rot": self.rigid_body_rot[:, self.handle_target_handle],
                "dof_pos": self.dof_pos,
                "dof_vel": self.dof_vel,
                "default_dof_pos": self.default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1)
            }
        )

    def _refresh_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._update_states()

    def compute_reward(self):
        # Synchronize with diablo2.py signature
        dummy_finger = torch.zeros_like(self.states["eef_pos"])
        
        target_pos_tensor = torch.tensor([self.target_pos_local.x, self.target_pos_local.y, self.target_pos_local.z], device=self.device).repeat(self.num_envs, 1)

        self.rew_buf[:], self.reset_buf[:] = compute_diablo_reward(
            self.reset_buf, self.progress_buf, self.actions,
            self.states["eef_pos"], self.states["handle_target_pos"],
            self.states["eef_rot"], self.states["handle_target_rot"],
            dummy_finger, dummy_finger, dummy_finger,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, 1.0, 1.0,
            1.0, self.action_penalty_scale, self.lift_reward_scale, self.orientation_reward_scale,
            self.max_episode_length,
            self.states["mug_pos"], self.states["mug_rot"], self.initial_object_z,
            target_pos_tensor
        )

    def compute_observations(self):
        self._refresh_tensors()
        delta = self.dof_upper_limits - self.dof_lower_limits
        obs_dof_pos = (self.states["dof_pos"] - self.dof_lower_limits) / (delta + 1e-6)
        obs_dof_vel = self.states["dof_vel"]

        # 計算相對位移向量 (手到杯柄)，這對「靠近」動作極其重要
        rel_pos = self.states["handle_target_pos"] - self.states["eef_pos"]

        self.obs_buf[:, 0] = self.progress_buf / self.max_episode_length
        self.obs_buf[:, 1 : 1 + self.num_dofs] = obs_dof_pos
        self.obs_buf[:, 1 + self.num_dofs : 1 + 2 * self.num_dofs] = obs_dof_vel
        
        start_idx = 1 + 2 * self.num_dofs
        self.obs_buf[:, start_idx : start_idx + 3] = self.states["mug_pos"]
        self.obs_buf[:, start_idx + 3 : start_idx + 7] = self.states["mug_rot"]
        
        # 關鍵：加入相對位移 (佔用 3 維)
        self.obs_buf[:, start_idx + 7 : start_idx + 10] = rel_pos
        self.obs_buf[:, start_idx + 10 : start_idx + 21] = self.actions[:, :11] 
        
        # 加入目標高度與俯仰角 (佔用最後 2 維)
        self.obs_buf[:, start_idx + 21] = self.target_height
        self.obs_buf[:, start_idx + 22] = self.target_pitch
        
        self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=0.0, neginf=0.0)
        return self.obs_buf

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        
        # --- 隨機化高度與傾斜目標 ---
        # 重置機身離地高度: 0.28m ~ 0.42m (中心值 0.35m)
        h_target = 0.35 + (torch.rand(num_resets, device=self.device) - 0.5) * 0.14
        p_target = (torch.rand(num_resets, device=self.device) - 0.5) * 0.4
        
        self.target_height[env_ids] = h_target - 0.35
        self.target_pitch[env_ids] = p_target

        # --- 逆運動學 (IK) 計算 ---
        # 參數: 輪徑 R=0.08, 腿長 l1=0.14, l2=0.14, Hip偏移=0.05
        R, l1, l2, offset = 0.12, 0.14, 0.14, 0.0
        L0 = (h_target - R - offset) / torch.cos(p_target)
        L0 = torch.clamp(L0, 0.12, 0.27)
        
        cos_alpha = (l1**2 + l2**2 - L0**2) / (2 * l1 * l2)
        alpha = torch.acos(torch.clamp(cos_alpha, -0.99, 0.99))
        target_knee = alpha - 3.14159 + 0.268
        target_hip = -p_target - target_knee / 2.0

        # --- 更新關節狀態 ---
        # 同時設定左腿 [0, 1] 與 右腿 [3, 4]
        self.dof_pos[env_ids, 0] = target_hip
        self.dof_pos[env_ids, 1] = target_knee
        self.dof_pos[env_ids, 3] = target_hip
        self.dof_pos[env_ids, 4] = target_knee
        self.dof_vel[env_ids, :] = 0.0
        
        # --- 更新機身狀態 (Root State) ---
        # 機身位移補償 (避免撞到桌子)
        safe_x_offset = - (h_target * torch.sin(p_target)) - 0.05
        self.root_state[self.diablo_actor_ids[env_ids], 0] = safe_x_offset
        self.root_state[self.diablo_actor_ids[env_ids], 1] = 0.0
        self.root_state[self.diablo_actor_ids[env_ids], 2] = h_target
        
        pitch_quat = quat_from_euler_xyz(torch.zeros_like(p_target), p_target, torch.zeros_like(p_target))
        self.root_state[self.diablo_actor_ids[env_ids], 3:7] = pitch_quat
        self.root_state[self.diablo_actor_ids[env_ids], 7:13] = 0.0

        # --- 同步到物理引擎 ---
        # 非常重要: 同時發送 DOF 與 Root 的索引更新
        multi_actor_indices = torch.cat([self.diablo_actor_ids[env_ids], self.object_actor_ids[env_ids], self.zone_actor_ids[env_ids]]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(multi_actor_indices), len(multi_actor_indices))
        
        # 強制同步 DOF 狀態
        diablo_indices = self.diablo_actor_ids[env_ids].to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(diablo_indices), len(diablo_indices))

        # --- 更新物體 (Mug) ---
        object_indices = self.object_actor_ids[env_ids].to(torch.int32)
        sample_mug_pos = torch.zeros(num_resets, 3, device=self.device)
        
        # --- 調整杯子重置位置：固定在右側偏移區間 ---
        x_noise = (torch.rand(num_resets, device=self.device) - 0.5) * 2.0
        x_offset = 0.02 
        sample_mug_pos[:, 0] = 0.20 + x_noise * 0.05
        # 讓杯子每次都在桌子中心點「右移」5cm ~ 15cm (機器人座標系 Y 軸負向)
        sample_mug_pos[:, 1] = self.table_surface_pos.y - (0.10 + torch.rand(num_resets, device=self.device) * 0.10)
        sample_mug_pos[:, 2] = self.initial_object_z
        self.root_state[object_indices, :3] = sample_mug_pos

        initial_mug_rot = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=self.device).unsqueeze(0).repeat(num_resets, 1)
        aa_rot = torch.zeros(num_resets, 3, device=self.device)
        aa_rot[:, 2] = np.pi + 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
        self.root_state[object_indices, 3:7] = quat_mul(axisangle2quat(aa_rot), initial_mug_rot)
        self.root_state[object_indices, 7:13] = 0.0
        # 更新 Zone Marker 位置 (顯示馬克杯重製範圍)
        self.root_state[self.zone_actor_ids[env_ids], 0] = 0.20
        self.root_state[self.zone_actor_ids[env_ids], 1] = self.table_surface_pos.y - 0.15
        self.root_state[self.zone_actor_ids[env_ids], 2] = self.table_surface_pos.z -0.02 # 貼在桌面
 

                # 同時重置機器人與杯子的 Root 狀態
        all_indices = torch.cat([self.diablo_actor_ids[env_ids], self.object_actor_ids[env_ids], self.zone_actor_ids[env_ids]]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone().to(self.device)
        
        eef_pos = self.states["eef_pos"]
        handle_pos = self.states["handle_target_pos"]
        eef_rot = self.states["eef_rot"]
        handle_rot = self.states["handle_target_rot"]

        align_dist = torch.linalg.norm(eef_pos - handle_pos, ord=2, dim=-1)
        gripper_forward_axis = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        handle_forward_target = torch.tensor([-1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        
        axis1 = tf_vector(eef_rot, gripper_forward_axis)
        axis2 = tf_vector(handle_rot, handle_forward_target)
        
        # 加上姿勢對齊計算，用於 pre_step 判定
        gripper_up_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        handle_up_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        axis3 = tf_vector(eef_rot, gripper_up_axis)
        axis4 = tf_vector(handle_rot, handle_up_axis)
        
        dot1 = torch.bmm(axis1.view(self.num_envs, 1, 3), axis2.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        dot2 = torch.bmm(axis3.view(self.num_envs, 1, 3), axis4.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)

        targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device)
        targets[:] = self.default_dof_pos

        # --- 腿部關節 IK 計算 (使用動態索引) ---
        leg_indices = self.dof_indices['legs']
        R, l1, l2, offset = 0.12, 0.14, 0.14, 0.0
        h_actual = 0.35 + self.target_height
        L0_curr = (h_actual - R - offset) / torch.cos(self.target_pitch)
        L0_curr = torch.clamp(L0_curr, 0.12, 0.27)
        cos_alpha_curr = (l1**2 + l2**2 - L0_curr**2) / (2 * l1 * l2)
        alpha_curr = torch.acos(torch.clamp(cos_alpha_curr, -0.99, 0.99))
        
        target_knee_ik = alpha_curr - 3.14159 + 0.268
        target_hip_ik = -self.target_pitch - target_knee_ik / 2.0

        # 將目標套用到正確的索引上
        targets[:, leg_indices[0]] = target_hip_ik
        targets[:, leg_indices[1]] = target_knee_ik
        targets[:, leg_indices[2]] = target_hip_ik
        targets[:, leg_indices[3]] = target_knee_ik


        arm_dof_indices = self.dof_indices["right_arm"]
        arm_actions = self.actions[:, :4]
        arm_targets = self.action_scale * arm_actions + self.default_dof_pos[arm_dof_indices]
        targets[:, arm_dof_indices] = arm_targets

        u_gripper = self.actions[:, 13]
        
        # --- 放寬抓取判定，利於初期學習 ---
        is_close = (align_dist <= 0.025) # 從 0.025 放寬到 0.05
        is_oriented = (dot1 < -0.7) & (dot2 > 0.7) # 從 0.9 放寬到 0.7
        is_ready_to_grasp = is_close 
        # & is_oriented

        
        should_close = is_ready_to_grasp & (u_gripper.view(-1) >= 0.0)

        gripper_dof_indices = self.dof_indices["right_gripper"]
        targets_fingers = torch.where(
            should_close.unsqueeze(1).expand(-1, 9),
            self.gripper_upper_limits.expand(self.num_envs, 9),
            self.gripper_lower_limits.expand(self.num_envs, 9)
        )
        targets[:, gripper_dof_indices] = targets_fingers        

        targets = tensor_clamp(targets, self.dof_lower_limits, self.dof_upper_limits)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()
        self.compute_reward()

# @torch.jit.script
def compute_diablo_reward(
    reset_buf, progress_buf, actions,
    eef_pos, handle_pos, eef_rot, handle_rot,
    finger_tip1_pos, finger_tip2_pos, finger_tip3_pos,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, lift_reward_scale, orientation_reward_scale,
    max_episode_length,
    object_pos, object_rot, initial_object_z,
    target_marker_pos
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float, float, Tensor, Tensor, float, Tensor) -> Tuple[Tensor, Tensor]

    # --- Stage 1: Reaching and Alignment Rewards (Matched to diablo2.py) ---
    d = torch.norm(eef_pos - handle_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.05, dist_reward * 2.0, dist_reward)

    gripper_forward_axis = torch.tensor([0.0, 0.0, -1.0], device=eef_pos.device).repeat(num_envs, 1)
    gripper_up_axis = torch.tensor([1.0, 0.0, 0.0], device=eef_pos.device).repeat(num_envs, 1)
    handle_inward_axis = torch.tensor([1.0, 0.0, 0.0], device=eef_pos.device).repeat(num_envs, 1)
    handle_up_axis = torch.tensor([0.0, 0.0, 1.0], device=eef_pos.device).repeat(num_envs, 1)

    axis1 = tf_vector(eef_rot, gripper_forward_axis)
    axis2 = tf_vector(handle_rot, handle_inward_axis)
    axis3 = tf_vector(eef_rot, gripper_up_axis)
    axis4 = tf_vector(handle_rot, handle_up_axis)

    dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    
    rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)


    # --- Stage 2: Grasping Attempt Reward (Matched to diablo2.py) ---
    gripper_close_action = (actions[:, 13] >= 0.0)
    is_oriented_for_grasp = (dot1 < -0.7) & (dot2 > 0.7)
    is_close_to_grasp = (d < 0.025) & is_oriented_for_grasp 

    grasp_attempt_reward = torch.where(is_close_to_grasp & gripper_close_action, torch.ones_like(rot_reward) * 0.5, torch.zeros_like(rot_reward))

    # --- Stage 3: Lifting Reward (Enhanced for guidance) ---
    object_height = object_pos[:, 2] - initial_object_z
    is_grasping = is_close_to_grasp & gripper_close_action

    # Continuous lift reward based on height
    lift_reward = torch.where(is_grasping, 15.0 * torch.clamp(object_height, min=0.0), torch.zeros_like(rot_reward))
    # Bonus for reaching specific heights
    lift_reward = torch.where(is_grasping & (object_height > 0.02), lift_reward + 2.0, lift_reward) 

    # --- Stage 4: Orientation Reward (Matched to diablo2.py) ---
    mug_up_vec = torch.tensor([0.0, 0.0, 1.0], device=eef_pos.device).repeat(num_envs, 1)
    world_mug_up = tf_vector(object_rot, mug_up_vec)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=eef_pos.device).repeat(num_envs, 1)
    
    # Mug upright reward
    mug_dot_up = torch.bmm(world_mug_up.view(num_envs, 1, 3), world_up.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    mug_orientation_reward = mug_dot_up * mug_dot_up
    
    # EEF x-axis upright reward (New requirement: grip_site x-axis points up)
    eef_x_axis_local = torch.tensor([1.0, 0.0, 0.0], device=eef_pos.device).repeat(num_envs, 1)
    world_eef_x = tf_vector(eef_rot, eef_x_axis_local)
    eef_dot_up = torch.bmm(world_eef_x.view(num_envs, 1, 3), world_up.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    eef_orientation_reward = torch.clamp(eef_dot_up, min=0.0) # Reward pointing up [0, 1]

    lift_condition = is_grasping & (object_height > 0.005)
    
    # Combined orientation reward
    orientation_reward = (mug_orientation_reward + eef_orientation_reward) * 0.5
    orientation_reward = torch.where(lift_condition, orientation_reward, torch.zeros_like(orientation_reward))

    # --- Stage 5: Target Reaching (Success & Proximity Condition) ---
    dist_to_target = torch.norm(object_pos - target_marker_pos, p=2, dim=-1)
    
    # Continuous proximity reward: higher when closer to target marker
    # This provides the necessary gradient to move towards the green sphere
    proximity_reward = torch.where(is_grasping, 5.0 / (1.0 + dist_to_target * 10.0), torch.zeros_like(rot_reward))
    
    is_success = (dist_to_target < 0.01) # 觸碰到參考點（1公分範圍內）
    success_reward = torch.where(is_success, torch.ones_like(rot_reward) * 20.0, torch.zeros_like(rot_reward))

    # --- Penalties ---
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # --- Total Reward ---
    rewards = dist_reward_scale * dist_reward \
        + rot_reward_scale * rot_reward \
        + grasp_attempt_reward \
        + lift_reward_scale * lift_reward \
        + orientation_reward_scale * orientation_reward \
        + proximity_reward \
        + success_reward \
        - action_penalty_scale * action_penalty

    # --- Resets ---
    # 成功觸碰參考點後重製
    reset_buf = torch.where(is_success, torch.ones_like(reset_buf), reset_buf)
    
    # 其他原有重製條件
    reset_buf = torch.where(object_height > 0.07, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(object_pos[:, 2] < 0.1, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf

# @torch.jit.script
def tf_vector(rot, vec):
    # type: (Tensor, Tensor) -> Tensor
    return quat_apply(rot, vec)
