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

        # 定義關節名稱，以便從 URDF 動態獲取索引，避免硬編碼錯誤
        self.right_arm_names = ["r_sho_pitch", "r_sho_roll", "r_el", "r_wrist"]
        self.right_gripper_names = [
            "r_index_base", "r_index_middle", "r_index_tip",
            "r_mid_base", "r_mid_middle", "r_mid_tip",
            "r_thumb_base", "r_thumb_middle", "r_thumb_tip"
        ]
        self.dof_indices = {}
        self.target_height = torch.zeros(self.cfg["env"]["numEnvs"], dtype=torch.float32, device=sim_device)
        self.target_pitch = torch.zeros(self.cfg["env"]["numEnvs"], dtype=torch.float32, device=sim_device)
        self.locked_leg_pos = torch.zeros((self.cfg["env"]["numEnvs"], 4), dtype=torch.float32, device=sim_device)

        # 用於持久化隨機位置的張量
        self.platform_pos_tensor = torch.zeros((self.cfg["env"]["numEnvs"], 3), dtype=torch.float32, device=sim_device)
        self.target_marker_pos_tensor = torch.zeros((self.cfg["env"]["numEnvs"], 3), dtype=torch.float32, device=sim_device)
        self.zone_marker_pos_tensor = torch.zeros((self.cfg["env"]["numEnvs"], 3), dtype=torch.float32, device=sim_device)

        # observation and action space
        # Actions: 4 right arm joints + 9 unused + 1 right gripper control = 14
        self.cfg["env"]["numActions"] = 14 
        self.cfg["env"]["numObservations"] = 98 # 增加平台位置 (3) 與相對位移 (3)

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

        mug_asset_file = "urdf/ycb/ycb_urdfs-main/ycb_assets/035_power_drill.urdf"  
        mug_asset_options = gymapi.AssetOptions()
        mug_asset_options.fix_base_link = False
        mug_asset_options.use_mesh_materials = True
        mug_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        mug_asset_options.override_com = True
        mug_asset_options.override_inertia = True
        mug_asset_options.vhacd_enabled = True
        mug_asset_options.vhacd_params = gymapi.VhacdParams()
        mug_asset_options.vhacd_params.resolution = 1000
        self.object_asset = self.gym.load_asset(self.sim, asset_root, mug_asset_file, mug_asset_options)
        self.num_object_bodies = self.gym.get_asset_rigid_body_count(self.object_asset)
        self.num_object_shapes = self.gym.get_asset_rigid_shape_count(self.object_asset)
        self._mug_height = 0.1 

        total_bodies = self.num_diablo_bodies + self.num_table_bodies + self.num_object_bodies
        total_shapes = self.num_diablo_shapes + self.num_table_shapes + self.num_object_shapes

        diablo_start_pose = gymapi.Transform()
        diablo_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.25) 
        diablo_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_stand_pos = gymapi.Vec3(0.25, 0.0, 0.35) # 調低桌子高度
        table_stand_pose = gymapi.Transform()
        table_stand_pose.p = table_stand_pos
        table_stand_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.table_surface_pos = table_stand_pos + gymapi.Vec3(0, 0, table_stand_height / 2)

        object_start_pose = gymapi.Transform()
        object_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        # 初始化為張量並填充預設值
        self.initial_object_z = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        default_z = self.table_surface_pos.z + self._mug_height / 2 + 0.002
        self.initial_object_z[:] = default_z

        # --- 新增漂浮小平台資產 ---
        platform_dims = gymapi.Vec3(0.1, 0.1, 0.01)
        platform_asset_options = gymapi.AssetOptions()
        platform_asset_options.fix_base_link = True
        platform_asset = self.gym.create_box(self.sim, platform_dims.x, platform_dims.y, platform_dims.z, platform_asset_options)

        self.envs = []
        self.actor_handles = []
        self.object_handles = []
        self.table_handles = []
        self.platform_handles = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, total_bodies + 3, total_shapes + 3, True)

            diablo_actor = self.gym.create_actor(env_ptr, diablo_asset, diablo_start_pose, "diablo", i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, diablo_actor, dof_props)
            self.actor_handles.append(diablo_actor)

            table_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_pose, "table", i, 1, 0)
            self.table_handles.append(table_actor)

            mug_start_pos_z = default_z
            object_start_pose.p = gymapi.Vec3(self.table_surface_pos.x, self.table_surface_pos.y + np.random.uniform(-0.25, 0.0), mug_start_pos_z)
            object_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0) 
            object_actor = self.gym.create_actor(env_ptr, self.object_asset, object_start_pose, "mug", i, 2, 0) 
            self.object_handles.append(object_actor)

            # --- 增加虛擬參考點 (Target Marker) ---
            target_marker_opts = gymapi.AssetOptions()
            target_marker_opts.fix_base_link = True
            target_marker_opts.disable_gravity = True
            target_marker_asset = self.gym.create_sphere(self.sim, 0.01, target_marker_opts)
            
            # 設定位置：使用 float 類型的 default_z 避免 Tensor 報錯
            self.target_pos_local = gymapi.Vec3(
                self.table_surface_pos.x,
                self.table_surface_pos.y - 0.10, 
                default_z + 0.05
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
            zone_marker_asset = self.gym.create_box(self.sim, 0.20, 0.15, 0.001, zone_marker_opts)
            
            # 初始位置預設在桌面上
            zone_pose = gymapi.Transform()
            zone_actor = self.gym.create_actor(env_ptr, zone_marker_asset, zone_pose, 'zone_marker', i, 3, 1)
            self.gym.set_rigid_body_color(env_ptr, zone_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 1.0)) # 藍色

            self.gym.set_rigid_body_color(env_ptr, target_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 1.0, 0.0)) # 綠色

            # --- 創建漂浮小平台 Actor ---
            platform_pose = gymapi.Transform()
            platform_pose.p = gymapi.Vec3(0.4, -0.2, 0.4)
            platform_actor = self.gym.create_actor(env_ptr, platform_asset, platform_pose, "platform", i, 5, 0)
            self.gym.set_rigid_body_color(env_ptr, platform_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.4, 0.1)) # 橘色
            self.platform_handles.append(platform_actor)

            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)

        self.diablo_actor_ids = torch.arange(0, self.num_envs * 6, 6, dtype=torch.int32, device=self.device)
        self.table_actor_ids = torch.arange(1, self.num_envs * 6, 6, dtype=torch.int32, device=self.device)
        self.object_actor_ids = torch.arange(2, self.num_envs * 6, 6, dtype=torch.int32, device=self.device)
        self.target_actor_ids = torch.arange(3, self.num_envs * 6, 6, dtype=torch.int32, device=self.device)
        self.zone_actor_ids = torch.arange(4, self.num_envs * 6, 6, dtype=torch.int32, device=self.device)
        self.platform_actor_ids = torch.arange(5, self.num_envs * 6, 6, dtype=torch.int32, device=self.device)

        self.eef_handle = self.gym.find_actor_rigid_body_index(self.envs[0], self.actor_handles[0], "panda_grip_site", gymapi.DOMAIN_ENV)
        self.handle_target_handle = self.gym.find_actor_rigid_body_index(self.envs[0], self.object_handles[0], "handle_target", gymapi.DOMAIN_ENV)
        
        print(f"EEF Local Index: {self.eef_handle}")
        print(f"Handle Target Local Index: {self.handle_target_handle}")

        self.init_data()

    def init_data(self):
        self._acquire_tensors()

        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float32, device=self.device)
        arm_default = np.radians([0, -60, -80, 0])
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
        # 使用 reset_idx 中儲存的隨機位置更新 root_state
        self.root_state[self.target_actor_ids, :3] = self.target_marker_pos_tensor
        self.root_state[self.zone_actor_ids, :3] = self.zone_marker_pos_tensor
        self.root_state[self.platform_actor_ids, :3] = self.platform_pos_tensor
        
        # 同步這些固定 Actor 的位置到模擬器
        all_fixed_indices = torch.cat([self.zone_actor_ids, self.target_actor_ids, self.platform_actor_ids])
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(all_fixed_indices), len(all_fixed_indices))

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
        
        # 傳入平台位置而非原本的 target_marker
        platform_pos = self.platform_pos_tensor

        self.rew_buf[:], self.reset_buf[:] = compute_diablo_reward(
            self.reset_buf, self.progress_buf, self.actions,
            self.states["eef_pos"], self.states["handle_target_pos"],
            self.states["eef_rot"], self.states["handle_target_rot"],
            dummy_finger, dummy_finger, dummy_finger,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, 1.0, 1.0,
            1.0, self.action_penalty_scale, self.lift_reward_scale, self.orientation_reward_scale,
            self.max_episode_length,
            self.states["mug_pos"], self.states["mug_rot"], self.initial_object_z,
            platform_pos,
            self.root_state[self.diablo_actor_ids, 3:7] # robot_rot
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
        
        # 加入目標高度與俯仰角 (佔用 2 維)
        self.obs_buf[:, start_idx + 21] = self.target_height
        self.obs_buf[:, start_idx + 22] = self.target_pitch
        
        # 新增：平台位置 (3) 與 物品到平台的位移 (3)
        rel_obj_to_plat = self.platform_pos_tensor - self.states["mug_pos"]
        self.obs_buf[:, start_idx + 23 : start_idx + 26] = self.platform_pos_tensor
        self.obs_buf[:, start_idx + 26 : start_idx + 29] = rel_obj_to_plat

        self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=0.0, neginf=0.0)
        return self.obs_buf

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        
        # --- IK-based RANDOM Reset: Logic adapted from visualize_leg_ik.py ---

        # 1. Define constants (R is updated based on visualization script)
        R, l1, l2 = 0.08, 0.14, 0.14
        max_leg_length = l1 + l2 # 0.28

        # 2. Randomize target height and pitch
        # Height range adjusted for new wheel radius R, with more obvious variation
        h_target = 0.23 + torch.rand(num_resets, device=self.device) * 0.12 # Range [0.23, 0.35]
        # Using more obvious pitch variation
        p_target = (torch.rand(num_resets, device=self.device) - 0.5) * 0.4 # Range [-0.2, 0.2]
        y_target = (torch.rand(num_resets, device=self.device) - 0.5) * 0.4 # Reduced random yaw range [-0.2, 0.2] rad

        # 3. Use IK to find the required leg configuration
        # Calculate required virtual leg length L0
        L0 = (h_target - R) / torch.cos(p_target)
        L0 = torch.clamp(L0, 0.01, max_leg_length)

        # 4. Calculate target joint angles using robust geometric derivation
        # 4.1 Knee angle (alpha is the internal angle at the knee)
        cos_alpha = (l1**2 + l2**2 - L0**2) / (2 * l1 * l2)
        cos_alpha = torch.clamp(cos_alpha, -1.0, 1.0)
        alpha = torch.acos(cos_alpha)
        target_knee = 3.14159 - alpha # q_knee = pi - alpha

        # 4.2 Hip angle (beta is the internal angle at the hip)
        # Since l1=l2, cos_beta simplifies from (l1^2 + L0^2 - l2^2)/(2*l1*L0)
        cos_beta = L0 / (2 * l1)
        cos_beta = torch.clamp(cos_beta, -1.0, 1.0)
        beta = torch.acos(cos_beta)
        # Total hip angle is angle of virtual leg (-p_target) minus internal angle beta.
        # The sign of beta determines which way the knee bends.
        target_hip = -p_target - beta

        # 4. Set and store the states
        self.target_height[env_ids] = h_target - 0.35 # For observation
        self.target_pitch[env_ids] = p_target      # For observation

        # Store locked angles for pre_physics_step
        leg_indices = self.dof_indices["legs"]
        self.locked_leg_pos[env_ids, 0] = target_hip
        self.locked_leg_pos[env_ids, 1] = target_knee
        self.locked_leg_pos[env_ids, 2] = target_hip
        self.locked_leg_pos[env_ids, 3] = target_knee
        
        # 5. Set initial DOF state
        self.dof_pos[env_ids, leg_indices[0]] = target_hip
        self.dof_pos[env_ids, leg_indices[1]] = target_knee
        self.dof_pos[env_ids, leg_indices[2]] = target_hip
        self.dof_pos[env_ids, leg_indices[3]] = target_knee
        
        # --- 強制將手臂與夾爪拉回預設安全位置，避免與杯子重疊 ---
        arm_and_gripper_indices = self.dof_indices["right_arm"] + self.dof_indices["right_gripper"]
        self.dof_pos[env_ids[:, None], arm_and_gripper_indices] = self.default_dof_pos[arm_and_gripper_indices]
        self.dof_vel[env_ids, :] = 0.0
        
        # 6. Set initial root state with corrective offset
        base_x_offset = L0 * torch.sin(p_target) * torch.cos(y_target)
        base_y_offset = L0 * torch.sin(p_target) * torch.sin(y_target)
        self.root_state[self.diablo_actor_ids[env_ids], 0] = base_x_offset
        self.root_state[self.diablo_actor_ids[env_ids], 1] = base_y_offset
        self.root_state[self.diablo_actor_ids[env_ids], 2] = h_target
        
        combined_quat = quat_from_euler_xyz(torch.zeros_like(p_target), p_target, y_target)
        self.root_state[self.diablo_actor_ids[env_ids], 3:7] = combined_quat
        self.root_state[self.diablo_actor_ids[env_ids], 7:13] = 0.0


        
        diablo_indices = self.diablo_actor_ids[env_ids].to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(diablo_indices), len(diablo_indices))

        object_indices = self.object_actor_ids[env_ids].to(torch.int32)
        sample_mug_pos = torch.zeros(num_resets, 3, device=self.device)
        
        x_noise = (torch.rand(num_resets, device=self.device) - 0.5) * 2.0
        sample_mug_pos[:, 0] = 0.20 + x_noise * 0.075
        sample_mug_pos[:, 1] = self.table_surface_pos.y - (0.075 + torch.rand(num_resets, device=self.device) * 0.15)
        sample_mug_pos[:, 2] = self.initial_object_z[env_ids]
        self.root_state[object_indices, :3] = sample_mug_pos

        initial_mug_rot = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=self.device).unsqueeze(0).repeat(num_resets, 1)
        aa_rot = torch.zeros(num_resets, 3, device=self.device)
        aa_rot[:, 2] = np.pi + 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
        self.root_state[object_indices, 3:7] = quat_mul(axisangle2quat(aa_rot), initial_mug_rot)
        self.root_state[object_indices, 7:13] = 0.0
        
        # 強制設定標記物坐標，確保不掉在地上
        self.target_marker_pos_tensor[env_ids, 0] = 0.25
        self.target_marker_pos_tensor[env_ids, 1] = self.table_surface_pos.y - 0.10
        self.target_marker_pos_tensor[env_ids, 2] = self.initial_object_z[env_ids] + 0.15
        
        self.zone_marker_pos_tensor[env_ids, 0] = 0.20
        self.zone_marker_pos_tensor[env_ids, 1] = self.table_surface_pos.y - 0.15
        self.zone_marker_pos_tensor[env_ids, 2] = 0.356
        
        # --- 初始隱藏漂浮小平台 (Z = -1.0) ---
        platform_indices = self.platform_actor_ids[env_ids].to(torch.int32)
        self.platform_pos_tensor[env_ids, 0] = 0.18
        self.platform_pos_tensor[env_ids, 1] = -0.30
        self.platform_pos_tensor[env_ids, 2] = -1.0
        
        self.root_state[platform_indices, :3] = self.platform_pos_tensor[env_ids]
        self.root_state[platform_indices, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(num_resets, 1)
        self.root_state[platform_indices, 7:13] = 0.0

        all_indices = torch.cat([
            self.diablo_actor_ids[env_ids], 
            self.object_actor_ids[env_ids], 
            self.target_actor_ids[env_ids], 
            self.zone_actor_ids[env_ids],
            self.platform_actor_ids[env_ids]
        ]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone().to(self.device)
        
        targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float32, device=self.device)
        targets[:] = self.default_dof_pos

        # --- Lock leg joints to the position set at reset ---
        leg_indices = self.dof_indices["legs"]
        locked_targets = torch.cat(
            (self.locked_leg_pos[:, 0].unsqueeze(1), self.locked_leg_pos[:, 1].unsqueeze(1),
             self.locked_leg_pos[:, 2].unsqueeze(1), self.locked_leg_pos[:, 3].unsqueeze(1)),
            dim=1
        )
        targets[:, leg_indices] = locked_targets

        # --- RL Agent Controls Arm ---
        arm_dof_indices = self.dof_indices["right_arm"]
        arm_actions = self.actions[:, :4]
        arm_targets = self.action_scale * arm_actions + self.default_dof_pos[arm_dof_indices]
        targets[:, arm_dof_indices] = arm_targets

        # --- Heuristic Gripper Control ---
        eef_pos = self.states["eef_pos"]
        handle_pos = self.states["handle_target_pos"]
        align_dist = torch.linalg.norm(eef_pos - handle_pos, ord=2, dim=-1)
        
        u_gripper = self.actions[:, 13]
        is_ready_to_grasp = (align_dist <= 0.025)
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

        # --- 新增：平台動態出現邏輯 ---
        # 檢測物體是否被抬高 (Height > 0.02m)
        object_heights = self.states["mug_pos"][:, 2] - self.initial_object_z
        # 這裡不強求 is_grasping，只要高度到了就視為觸發出現
        should_appear = (object_heights > 0.02)
        
        # 將滿足條件的環境平台 Z 軸設為固定目標高度 (桌面 + 0.03m)
        target_z = self.table_surface_pos.z + 0.03
        self.platform_pos_tensor[:, 2] = torch.where(
            should_appear, 
            torch.ones_like(self.platform_pos_tensor[:, 2]) * target_z, 
            self.platform_pos_tensor[:, 2]
        )

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()
        self.compute_reward()

@torch.jit.script
def compute_diablo_reward(
    reset_buf, progress_buf, actions,
    eef_pos, handle_pos, eef_rot, handle_rot,
    finger_tip1_pos, finger_tip2_pos, finger_tip3_pos,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, lift_reward_scale, orientation_reward_scale,
    max_episode_length,
    object_pos, object_rot, initial_object_z,
    platform_pos,
    robot_rot
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float, float, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    # --- Stage 1: Reaching and Alignment (靠近把手) ---
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

    # --- Stage 2: Grasping (抓取) ---
    gripper_close_action = (actions[:, 13] >= 0.0)
    is_oriented_for_grasp = (dot1 < -0.7) & (dot2 > 0.7)
    is_close_to_grasp = (d < 0.025) & is_oriented_for_grasp 
    # 提高抓取意圖獎勵
    grasp_attempt_reward = torch.where(is_close_to_grasp & gripper_close_action, torch.ones_like(rot_reward) * 5.0, torch.zeros_like(rot_reward))

    # --- Stage 3: Lifting & Transport (抬升與搬運) ---
    object_height = object_pos[:, 2] - initial_object_z
    is_grasping = is_close_to_grasp & gripper_close_action
    is_lifted = (object_height > 0.03)
    dist_xy_to_platform = torch.norm(object_pos[:, :2] - platform_pos[:, :2], p=2, dim=-1)
    # 縮小對準範圍：要求更精確的中心對準 (2.5cm)
    is_over_platform = (dist_xy_to_platform < 0.025) 
    
    platform_surface_z = platform_pos[:, 2] + 0.005
    object_bottom_z = object_pos[:, 2] - 0.05
    dist_z_to_platform = torch.abs(object_bottom_z - platform_surface_z)
    # 同時要求更高的 Z 軸精度 (1.5cm)
    is_on_platform = is_over_platform & (dist_z_to_platform < 0.015)

    # --- Orientation (基礎計算) ---
    mug_up_vec = torch.tensor([0.0, 0.0, 1.0], device=eef_pos.device).repeat(num_envs, 1)
    world_mug_up = tf_vector(object_rot, mug_up_vec)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=eef_pos.device).repeat(num_envs, 1)
    mug_dot_up = torch.bmm(world_mug_up.view(num_envs, 1, 3), world_up.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    
    # 強制垂直判定：只有當 mug_dot_up > 0.85 (約 30度以內) 才算「基本垂直」
    is_upright = (mug_dot_up > 0.85)

    # --- 階段鎖定 (Stage Latching) 邏輯 ---
    # 1. 一旦舉起，靠近獎勵鎖定為常數
    dist_reward = torch.where(is_lifted, torch.ones_like(dist_reward) * 2.0, dist_reward)
    rot_reward = torch.where(is_lifted, torch.ones_like(rot_reward) * 0.5, rot_reward)

    # 2. 抬升獎勵：加上「必須垂直」的要求，否則不予鎖定
    target_lift = 0.06
    lift_reward = torch.where(is_grasping, 100.0 * torch.clamp(object_height, min=0.0, max=target_lift), torch.zeros_like(rot_reward))
    lift_reward = torch.where(is_over_platform & is_upright, torch.ones_like(lift_reward) * 100.0 * target_lift, lift_reward)

    # 3. 搬運獎勵：如果物品不垂直，搬運獎勵大幅縮減，防止刷分
    transport_reward = torch.where(is_grasping & is_lifted, 180.0 * torch.exp(-5.0 * dist_xy_to_platform), torch.zeros_like(rot_reward))
    transport_reward = torch.where(~is_upright, transport_reward * 0.1, transport_reward) # 懲罰橫躺搬運
    transport_reward = torch.where(is_on_platform & is_upright, torch.ones_like(transport_reward) * 180.0, transport_reward)

    # 4. 姿態獎勵：權重提高，並加入「如果倒下就歸零」的硬限制
    current_ori_reward = torch.pow(torch.clamp(mug_dot_up, min=0.0), 8) * 100.0 # 提高次方與權重
    orientation_reward = torch.where(is_on_platform & is_upright, torch.ones_like(rot_reward) * 100.0, 
                                     torch.where(object_height > 0.01, current_ori_reward, torch.zeros_like(rot_reward)))

    # 5. 放置獎勵：加上「垂直」門檻，否則無法進入鎖定狀態
    current_place_reward = 200.0 / (1.0 + dist_z_to_platform * 25.0 + dist_xy_to_platform * 40.0)
    placement_reward = torch.where(is_on_platform & is_upright, torch.ones_like(rot_reward) * 200.0,
                                   torch.where(is_grasping & (dist_xy_to_platform < 0.06), current_place_reward, torch.zeros_like(rot_reward)))

    # 高度懲罰：僅在非成功狀態下生效
    lift_penalty = torch.where(is_grasping & (object_height > target_lift + 0.04), 500.0 * (object_height - (target_lift + 0.04)), torch.zeros_like(rot_reward))

    # 時間懲罰
    time_penalty = torch.ones_like(rot_reward) * 1.5

    # --- Stage 5: Release & Success (放手與成功) ---
    gripper_open = (actions[:, 13] < -0.1) 
    eef_dist_to_obj = torch.norm(eef_pos - object_pos, p=2, dim=-1)
    
    # 1. 讓放手動作本身具有極高價值
    is_releasing = is_on_platform & gripper_open
    release_reward = torch.where(is_releasing, torch.ones_like(rot_reward) * 500.0, torch.zeros_like(rot_reward))

    # 2. 強化撤離獎勵：改為非線性梯度，建立爆炸性的「排斥場」
    # 距離越遠分越高，且使用平方項讓「遠離」的動作獲得急劇增加的分數
    # 當距離達到 15cm (0.15) 時，此項加分約為 2250 分 (100000 * 0.15^2)
    current_retreat_dist = torch.clamp(eef_dist_to_obj, max=0.15)
    retreat_reward = torch.where(is_releasing, torch.pow(current_retreat_dist, 2) * 100000.0, torch.zeros_like(rot_reward))

    # 3. 最終成功大獎：提高到 2500 分，誘使 Agent 衝刺過線
    # 要求手部明確撤離物品 (12cm 以上)
    is_success = is_on_platform & (gripper_open & (eef_dist_to_obj > 0.12))
    success_reward = torch.where(is_success, torch.ones_like(rot_reward) * 2500.0, torch.zeros_like(rot_reward))
    
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # --- Total Reward ---
    rewards = dist_reward_scale * dist_reward \
        + rot_reward_scale * rot_reward \
        + grasp_attempt_reward \
        + lift_reward_scale * lift_reward \
        + transport_reward \
        + placement_reward \
        + release_reward \
        + retreat_reward \
        + orientation_reward \
        + success_reward \
        - lift_penalty \
        - time_penalty \
        - action_penalty_scale * action_penalty

    # --- Resets ---
    reset_buf = torch.where(is_success, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    # 如果物品掉下桌面且不在平台上
    reset_buf = torch.where((object_pos[:, 2] < 0.2) & (~is_on_platform), torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf

@torch.jit.script
def tf_vector(rot, vec):
    # type: (Tensor, Tensor) -> Tensor
    return quat_apply(rot, vec)
