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
from isaacgymenvs.utils import isaacgym_utils
from isaacgymenvs.utils.torch_jit_utils import *


class diablo2(VecTask):
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
        self.diablo_dof_noise = self.cfg["env"]["diabloDofNoise"]

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {
            "cartesian",
            "joint",
        }, "Invalid control type specified. Must be one of: {cartesian, joint}"

        # observation and action space
        # 觀察空間：1 (progress) + 28 (dof_pos) + 28 (dof_vel) + 3 (target_pos) + 4 (target_rot) + 14 (actions) = 78
        self.cfg["env"]["numObservations"] = 78

        if self.control_type == "joint":
            # 動作空間：只包含右手關節（15-27，共13個）+ gripper（1個）
            # 排除左手關節（3-14）和頭部（0-2，會在 pre_physics_step 中固定為0）
            # 右手關節索引：15-27（13個關節）
            self.cfg["env"]["numActions"] = 14  # 13個右手關節 + 1個gripper控制
            self.right_arm_dof_indices = list(range(15, 28))  # 右手關節索引 15-27
        elif self.control_type == "cartesian":
            self.cfg["env"]["numActions"] = 3

        self.actions = torch.zeros(
            (self.cfg["env"]["numEnvs"], self.cfg["env"]["numActions"]),
            dtype=torch.float32,
            device=sim_device,
        )
        self.prev_actions = torch.zeros_like(self.actions)

        self._action_scale = self.cfg["env"]["actionScale"]
        self._dof_vel_scale = self.cfg["env"]["dofVelocityScale"]

        # Rewards
        self.dist_reward_scale = self.cfg["env"]["rewards"].get("distRewardScale", 1.0)
        self.rot_reward_scale = self.cfg["env"]["rewards"].get("rotRewardScale", 1.0)
        self.around_handle_reward_scale = self.cfg["env"]["rewards"].get("aroundHandleRewardScale", 1.0)
        self.open_reward_scale = self.cfg["env"]["rewards"].get("openRewardScale", 1.0)
        self.finger_dist_reward_scale = self.cfg["env"]["rewards"].get("fingerDistRewardScale", 1.0)
        self.action_penalty_scale = self.cfg["env"]["rewards"].get("actionPenaltyScale", 1.0)
        self.lift_reward_scale = self.cfg["env"]["rewards"].get("liftRewardScale", 1.0)
        self.orientation_reward_scale = self.cfg["env"]["rewards"].get("orientationRewardScale", 1.0)


        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        # Values to be filled in at runtime
        self.states = {}

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums ={"pos_err_penalty_": torch_zeros(), "pos_err_tanh_": torch_zeros(),"action_rate_penalty_": torch_zeros(), "joint_vel_penalty_": torch_zeros()}
        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

        if not self.headless:
            self._init_debug()

    

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

    def _create_assets(self):
        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.cfg["env"]["asset"]["assetRoot"],
        )
        # asset_root2 = os.path.join(
        #     os.path.dirname(os.path.abspath(__file__)),
        #     self.cfg["env"]["asset"]["assetRoot2"],
        # )
        diablo_asset_file = self.cfg["env"]["asset"]["assetFileNamediablo"]
        mug_asset_file = "/home/erc/isaacgym/assets/urdf/ycb/025_mug/025_mug.urdf"
        # diablo asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.armature = 0.01
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        # asset_options.vhacd_enabled = False
        diablo_asset = self.gym.load_asset(
            self.sim, asset_root, diablo_asset_file, asset_options
        )
        num_dofs = self.gym.get_asset_dof_count(diablo_asset)  # self.gym.get_asset_dof_count(diablo_asset)
        print("Total DOFs:", num_dofs)
        for i in range(num_dofs):
            dof_name = self.gym.get_asset_dof_name(diablo_asset, i)
            print(f"DOF {i}: {dof_name}")        
        # table asset
        self._table_thickness = self.cfg["env"]["asset"]["tableThickness"]
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(
            self.sim, *[0.5, 0.5, self._table_thickness], asset_options
        )

        # table stand asset
        self._table_stand_height = self.cfg["env"]["asset"]["tableStandHeight"]
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_stand_asset = self.gym.create_box(
            self.sim, *[0.3, 0.5, self._table_stand_height], asset_options
        )
        print("table stand height:", self._table_stand_height)
        
        # mug asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 1000
        mug_asset = self.gym.load_asset(self.sim, "", mug_asset_file, asset_options)
        self._mug_height = 0.1  # TODO: Set the correct mug height
        

        
        return {
            "diablo": diablo_asset,
            "table": table_asset,
            "table_stand": table_stand_asset,
            "mug": mug_asset,
        }

    def _set_diablo_dof_props(self, diablo_asset):
        # 獲取剛體和 DOF 數量
        self.rigid_body_dict_diablo = self.gym.get_asset_rigid_body_dict(diablo_asset)
        self.num_diablo_bodies = self.gym.get_asset_rigid_body_count(diablo_asset)
        self.num_diablo_dofs = self.gym.get_asset_dof_count(diablo_asset)
        diablo_dof_props = self.gym.get_asset_dof_properties(diablo_asset)

        # 打印剛體和 DOF 數量
        print("num diablo bodies: ", self.num_diablo_bodies)
        print("num diablo dofs: ", self.num_diablo_dofs)
        
        # 獲取 DOF 名稱並打印
        self.dof_names = self.gym.get_asset_dof_names(diablo_asset)
        # print("All DOF names: ", self.dof_names)
        diablo_dof_stiffness = torch.tensor(
            [16.08461, 1.1578, 15, 20, 13.82181, 4.54741, 1, 1, 1, 1, 1, 1, 1, 1, 1, 15, 20, 13.82181, 4.54741, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            dtype=torch.float32,
            device=self.device,
        )
        diablo_dof_damping = torch.tensor(
            [0.00643, 0.00046, 0.3, 0.4, 0.1, 0.00182, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.3, 0.4, 0.1, 0.00182,  0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001 ],
            dtype=torch.float,
            device=self.device,
        )

        self.diablo_dof_lower_limits = []
        self.diablo_dof_upper_limits = []

        for i in range(self.num_diablo_dofs):
            diablo_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                diablo_dof_props["stiffness"][i] = diablo_dof_stiffness[i]
                diablo_dof_props["damping"][i] = diablo_dof_damping[i]
            else:
                diablo_dof_props["stiffness"][i] = 7000.0
                diablo_dof_props["damping"][i] = 50.0
            
            # 增加左手關節的 stiffness 以確保它們保持固定
            # 根據你的設置，左手關節索引是 3-14
            if i >= 3 and i < 15:  # 左手關節範圍
                if self.physics_engine == gymapi.SIM_PHYSX:
                    diablo_dof_props["stiffness"][i] = 1000.0  # 提高 stiffness 確保固定
                    diablo_dof_props["damping"][i] = 100.0  # 提高 damping 減少震盪
                else:
                    diablo_dof_props["stiffness"][i] = 10000.0
                    diablo_dof_props["damping"][i] = 500.0

            self.diablo_dof_lower_limits.append(diablo_dof_props["lower"][i])
            self.diablo_dof_upper_limits.append(diablo_dof_props["upper"][i])

        self.diablo_dof_lower_limits = to_torch(
            self.diablo_dof_lower_limits, device=self.device
        )
        self.diablo_dof_upper_limits = to_torch(    
            self.diablo_dof_upper_limits, device=self.device
        )
        self.diablo_dof_speed_scales = torch.ones_like(
            self.diablo_dof_lower_limits
        )
        
        self.diablo_dof_speed_scales[[19, 20, 21, 22, 23, 24, 25, 26, 27]] = 0.1
        diablo_dof_props["effort"][19] = 200
        diablo_dof_props["effort"][20] = 200
        diablo_dof_props["effort"][21] = 200
        diablo_dof_props["effort"][22] = 200
        diablo_dof_props["effort"][23] = 200
        diablo_dof_props["effort"][24] = 200
        diablo_dof_props["effort"][25] = 200
        diablo_dof_props["effort"][26] = 200
        diablo_dof_props["effort"][27] = 200

        return diablo_dof_props
    
  

    def _init_start_poses(self):
        # start pose for franka
        diablo_start_pose = gymapi.Transform()
        diablo_start_pose.p = gymapi.Vec3(
            -0.15,
            0.0,
            self._table_thickness + 0.005,
        )
        diablo_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # start pose for table
        table_start_pose = gymapi.Transform()
        table_pos = [0.0, 0.0, 0.13]
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array(
            [0, 0, self._table_thickness / 2]
        )

        # start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_pos = [
            0.1,
            0.0,
            0.28,
        ]
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_stand_surface_pos = np.array(table_stand_pos) + np.array(
            [0, -0.15, self._table_stand_height / 2]
        )

        print("table stand surface pos:", self._table_stand_surface_pos)

        # start pose for mug (放在 table stand 上)
        mug_start_pose = gymapi.Transform()
        # 使用與 cube 相同的計算方式，加一個足夠的偏移避免穿透
        # 偏移量需要足夠大以避免碰撞檢測時的穿透問題
        self._mug_start_pos = self._table_stand_surface_pos + np.array(
            [0.0, 0.0, self._mug_height / 2 + 0.002]
        )
        mug_start_pose.p = gymapi.Vec3(*self._mug_start_pos)
        mug_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
        self.initial_mug_pos_z = self._mug_start_pos[2]


        return {
            "diablo": diablo_start_pose,
            "table": table_start_pose,
            "table_stand": table_stand_start_pose,
            "mug": mug_start_pose,
        }

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        ##########################################################
        # Create all assets
        ##########################################################
        assets_dict = self._create_assets()
        diablo_asset = assets_dict["diablo"]
        table_asset = assets_dict["table"]
        table_stand_asset = assets_dict["table_stand"]
        mug_asset = assets_dict["mug"]

        self._total_assets = len(assets_dict)

        ##########################################################
        # Set up franka dof properties
        ##########################################################
        diablo_dof_props = self._set_diablo_dof_props(diablo_asset)

        ##########################################################
        # Define start poses
        ##########################################################
        start_poses_dict = self._init_start_poses()
        diablo_start_pose = start_poses_dict["diablo"]
        table_start_pose = start_poses_dict["table"]
        table_stand_start_pose = start_poses_dict["table_stand"]
        mug_start_pose = start_poses_dict["mug"]

        # compute aggregate size
        num_diablo_bodies = self.gym.get_asset_rigid_body_count(diablo_asset)
        num_diablo_shapes = self.gym.get_asset_rigid_shape_count(diablo_asset)
        max_agg_bodies = (
            num_diablo_bodies + 3
        )  # 1 for table, 1 table stand, 1 mug
        max_agg_shapes = (
            num_diablo_shapes + 3
        )  # 1 for table, 1 table stand, 1 mug

        self.envs = []
        self.diablos = []
        self.mugs = []

        indexes_sim_diablo = []
        indexes_sim_mug = []

        ##########################################################
        # Create environments
        ##########################################################
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            self.gym.begin_aggregate(
                env_ptr, max_agg_bodies, max_agg_shapes, True
            )

            # Create franka
            diablo_actor = self.gym.create_actor(
                env_ptr, diablo_asset, diablo_start_pose, "diablo", i, -1, 0
            )
            self.gym.set_actor_dof_properties(
                env_ptr, diablo_actor, diablo_dof_props
            )
            indexes_sim_diablo.append(
                self.gym.get_actor_index(
                    env_ptr, diablo_actor, gymapi.DOMAIN_SIM
                )
            )

            # Create table
            self.gym.create_actor(
                env_ptr, table_asset, table_start_pose, "table", i, 1, 0
            )
            self.gym.create_actor(
                env_ptr,
                table_stand_asset,
                table_stand_start_pose,
                "table_stand", i,1,0,
            )

            # Create mug
            mug_actor = self.gym.create_actor(
                env_ptr, mug_asset, mug_start_pose, "mug", i, 2, 0
            )
            self.gym.set_rigid_body_color(
                env_ptr,
                mug_actor,
                0,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(1.0, 0.0, 0.0),
            )
            indexes_sim_mug.append(
                self.gym.get_actor_index(
                    env_ptr, mug_actor, gymapi.DOMAIN_SIM
                )
            )

            self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.diablos.append(diablo_actor)
            self.mugs.append(mug_actor)

        self.indexes_sim_diablo = torch.tensor(
            indexes_sim_diablo, dtype=torch.int32, device=self.device
        )
        self.indexes_sim_mug = torch.tensor(
            indexes_sim_mug, dtype=torch.int32, device=self.device
        )

        # Setup data
        self.init_data()


    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        diablo_handle = self.diablos[0]
        mug_handle = self.gym.find_actor_handle(env_ptr, "mug")
        handle_target_handle = self.gym.find_actor_rigid_body_handle(env_ptr, mug_handle, "handle_target")

        if handle_target_handle == -1:
            print("****** WARNING: handle_target not found. Make sure it is defined in the URDF. ******")

        print("env_ptr:", env_ptr)
        print("diablo_handle:", diablo_handle)
        self.handles = {
            # # Franka
            # "hand": self.gym.find_actor_rigid_body_handle(
            #     env_ptr, diablo_handle, "panda_hand"
            # ),

            "r_index_tip": self.gym.find_actor_rigid_body_handle(
                env_ptr, diablo_handle, "r_index_tip"
            ),
            "r_mid_tip": self.gym.find_actor_rigid_body_handle(
                env_ptr, diablo_handle, "r_mid_tip"
            ),
            "r_thumb_tip": self.gym.find_actor_rigid_body_handle(
                env_ptr, diablo_handle, "r_thumb_tip"
            ),
            "grip_site": self.gym.find_actor_rigid_body_handle(
                env_ptr, diablo_handle, "panda_grip_site"
            ),
            # Mug
            "mug": mug_handle,
            "handle_target": handle_target_handle,
        }



        # Right arm defaults

        # self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        self.diablo_default_dof_pos = torch.tensor(
            np.radians([0, 0, 0, -57, -80, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0, 0, -57, -80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            device=self.device,
            dtype=torch.float32,
        )
        # # Right arm defaults
        # self.diablo_default_dof_pos = torch.tensor(
        #     np.radians([0 , -57, -80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        #     device=self.device,
        #     dtype=torch.float32,
        # )
        self.diablo_dof_targets = torch.zeros(
            (self.num_envs, self.num_diablo_dofs),
            dtype=torch.float32,
            device=self.device,
        )
        # Initialize actions

        self._pos_control = torch.zeros(        
            (self.num_envs, self.num_diablo_dofs), dtype=torch.float32, device=self.device
        )
        self._effort_control = torch.zeros_like(self._pos_control)


        # Initialize control
        self._arm_control = self._effort_control[:, :18]

        self._gripper_control = self._pos_control[:, 18:27]
        self.u_fingers = torch.zeros((self.num_envs, 9), dtype=torch.float32, device=self.device)


        # Initialize indices
        # self._global_indices = torch.arange(self.num_envs * 5, dtype=torch.int32,
        #                                    device=self.device).view(self.num_envs, -1)

        # self.diablo_dof_gripper = torch.zeros(
        #     (self.num_envs, 2), dtype=torch.float32, device=self.device
        # )    


        # Setup tensor buffers and views: roots, DOFs, rigid bodies.
        root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(
            self.sim
        )


        self.log_reward_tensor = torch.zeros(     
            (4, self.num_envs), dtype=torch.float32, device=self.device
        )

        if self.control_type == "cartesian":
            jacobian_tensor = self.gym.acquire_jacobian_tensor(
                self.sim, "diablo"
            )

        self.root_state = gymtorch.wrap_tensor(root_state_tensor)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor)

        if self.control_type == "cartesian":
            self.jacobian = gymtorch.wrap_tensor(jacobian_tensor)

        # Root states
        self.root_pos = self.root_state[:, :3].view(self.num_envs, -1, 3)
        self.root_rot = self.root_state[:, 3:7].view(self.num_envs, -1, 4)
        self.root_vel_lin = self.root_state[:, 7:10].view(self.num_envs, -1, 3)
        self.root_vel_ang = self.root_state[:, 10:13].view(
            self.num_envs, -1, 3
        )

        # DoF states
        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[..., 1]

        # Rigid body states
        self.rigid_body_pos = self.rigid_body_state[:, :3].view(
            self.num_envs, -1, 3
        )
        self.rigid_body_rot = self.rigid_body_state[:, 3:7].view(
            self.num_envs, -1, 4
        )
        self.rigid_body_vel_lin = self.rigid_body_state[:, 7:10].view(
            self.num_envs, -1, 3
        )
        self.rigid_body_vel_ang = self.rigid_body_state[:, 10:13].view(
            self.num_envs, -1, 3
        )

        if self.control_type == "cartesian":
            hand_joint_index = self.gym.get_actor_joint_dict(
                env_ptr, diablo_handle
            )["r_wrist"]
            self.jacobian_eef = self.jacobian[:, hand_joint_index, :, :]

        self._global_indices = torch.arange(
            self.num_envs * self._total_assets,
            dtype=torch.int32,
            device=self.device,
        ).view(self.num_envs, -1)





    def _init_debug(self):
        # Focus viewer's camera on the first environment.
        self.flag_camera_look_at = self.cfg["render"].get(
            "enableCameraLookAtEnv", False
        )
        self.i_look_at_env = self.cfg["render"].get("cameraLookAtEnvId", 0)
        self.debug_cam_pos = gymapi.Vec3(*self.cfg["render"]["cameraPosition"])
        self.debug_cam_target = gymapi.Vec3(
            *self.cfg["render"]["cameraTarget"]
        )
        self.flag_debug_vis = self.cfg["render"].get("enableDebugVis", False)
        self.gym.viewer_camera_look_at(
            self.viewer,
            self.envs[self.i_look_at_env],
            self.debug_cam_pos,
            self.debug_cam_target,
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_T, "camera_look_at"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_Y, "env_prev"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_U, "env_next"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_R, "reset_envs"
        )
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer, gymapi.KEY_P, "debug_vis"
        )

    def keyboard(self, event):
        if event.action == "camera_look_at" and event.value > 0:
            self.flag_camera_look_at = not self.flag_camera_look_at
        elif event.action == "env_prev" and event.value > 0:
            self.i_look_at_env = max(0, self.i_look_at_env - 1)
            self.update_debug_camera()
        elif event.action == "env_next" and event.value > 0:
            self.i_look_at_env = min(self.i_look_at_env + 1, self.num_envs - 1)
            self.update_debug_camera()
        elif event.action == "reset_envs" and event.value > 0:
            self.reset_buf[:] = 1
        elif event.action == "debug_vis" and event.value > 0:
            self.gym.clear_lines(self.viewer)
            self.flag_debug_vis = not self.flag_debug_vis

    def viewer_update(self):
        if self.flag_debug_vis:
            self._draw_debug_vis()

    def update_debug_camera(self):
        if not self.flag_camera_look_at:
            return

        self.gym.viewer_camera_look_at(
            self.viewer,
            self.envs[self.i_look_at_env],
            self.debug_cam_pos,
            self.debug_cam_target,
        )        
    
    def _update_states(self):
        self.states.update(
            {
                # Franka
                "dof_pos": self.dof_pos[:, :],
                # "dof_gripper": self.dof_pos[:, -9:],
                "dof_vel": self.dof_vel[:, :],
                # End effector
                "eef_pos": self.rigid_body_pos[:, self.handles["grip_site"]],
                "r_index_tip_pos": self.rigid_body_pos[:, self.handles["r_index_tip"]],
                "r_mid_tip_pos": self.rigid_body_pos[:, self.handles["r_mid_tip"]],
                "r_thumb_tip_pos": self.rigid_body_pos[:, self.handles["r_thumb_tip"]],
                "eef_rot": self.rigid_body_rot[:, self.handles["grip_site"]],
                "eef_vel_lin": self.rigid_body_vel_lin[
                    :, self.handles["grip_site"]
                ],
                "ee_vel_ang": self.rigid_body_vel_ang[
                    :, self.handles["grip_site"]
                ],
                # Mug
                "mug_pos": self.root_pos[:, self.handles["mug"]],
                "mug_rot": self.root_rot[:, self.handles["mug"]],
                "handle_target_pos": self.rigid_body_pos[:, self.handles["handle_target"]],
                "handle_target_rot": self.rigid_body_rot[:, self.handles["handle_target"]],
            }
        )

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.control_type == "cartesian":
            self.gym.refresh_jacobian_tensors(self.sim)

        # Refresh states
        self._update_states()

    def _draw_debug_vis(self):
        """Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
        """
        # draw height lines
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        visualize_sphere = {
            "eef_pos": gymutil.WireframeSphereGeometry(
                0.02,
                8,
                8,
                None,
                color=(0, 1, 0),
            ),
            "mug_pos": gymutil.WireframeSphereGeometry(
                0.02,
                8,
                8,
                None,
                color=(1, 1, 0),
            ),
        }

        for i in range(self.num_envs):
            for pose, geom in visualize_sphere.items():
                x, y, z = self.states[pose][i].cpu().numpy()
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(
                    geom, self.gym, self.viewer, self.envs[i], sphere_pose
                )

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_diablo_reward(
            self.reset_buf, self.progress_buf, self.actions,
            self.states["eef_pos"], self.states["handle_target_pos"],
            self.states["eef_rot"], self.states["handle_target_rot"],
            self.states["r_index_tip_pos"], self.states["r_mid_tip_pos"], self.states["r_thumb_tip_pos"],
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale,
            self.around_handle_reward_scale, self.open_reward_scale,
            self.finger_dist_reward_scale, self.action_penalty_scale, self.lift_reward_scale,
            self.orientation_reward_scale,
            self.max_episode_length, self.states["mug_pos"], self.states["mug_rot"], self.initial_mug_pos_z
        )

    def compute_observations(self):
        self._refresh()

        dof_pos_scaled = (
            2.0
            * (self.states["dof_pos"] - self.diablo_dof_lower_limits)
            / (self.diablo_dof_upper_limits - self.diablo_dof_lower_limits)
            - 1.0
        )
        dof_vel_scaled = self.states["dof_vel"] * self._dof_vel_scale

        self.obs_buf[:, 0] = self.progress_buf / self.max_episode_length
        self.obs_buf[:, 1:29] = dof_pos_scaled[:, :28]
        self.obs_buf[:, 29:57] = dof_vel_scaled[:, :28]
        self.obs_buf[:, 57:60] = self.states["mug_pos"]
        self.obs_buf[:, 60:64] = self.states["mug_rot"]
        
        # 動作空間現在是 14 維（13個右手關節 + 1個gripper）
        self.obs_buf[:, 64:78] = self.actions

        return self.obs_buf
    

    def reset_idx(self, env_ids):
        ##################################################################
        # Reset diablo
        ##################################################################
        num_resets = len(env_ids)
        dof_noise = torch.rand(
            (num_resets, self.num_diablo_dofs), device=self.device
        )
        pos = self.diablo_default_dof_pos.unsqueeze(
            0
        ) + self.diablo_dof_noise * 2.0 * (dof_noise - 0.5)
        pos = tensor_clamp(
            pos, self.diablo_dof_lower_limits, self.diablo_dof_upper_limits
        )
        # print("env_ids: ", env_ids)
        # Overwrite gripper init pos
        # (no noise since these are always position controlled)
        pos[:, -9:] = self.diablo_default_dof_pos[-9:]
        # pos[:, -2:] = 0.0 # close gripper
        
        # 固定左手關節到預設位置（確保 reset 時左手不會動）
        # 根據你的設置，左手關節索引應該是 3-14（包含手臂和手指）
        # 如果範圍不同，請根據實際 DOF 索引調整
        pos[:, 3:15] = self.diablo_default_dof_pos[3:15]  # 左手關節固定到預設值

        self.diablo_dof_targets[env_ids, :] = pos[:]
        self._effort_control[env_ids, :] = torch.zeros_like(pos)
        
        self.dof_pos[env_ids, :] = pos
        self.dof_vel[env_ids, :] = 0.0
        # print("diablo_dof_targets shape : ", self.diablo_dof_targets.shape)
        # print("dof_pos shape : ", pos.shape)

        indexes = self.indexes_sim_diablo[env_ids]
        # multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.diablo_dof_targets),
            gymtorch.unwrap_tensor(indexes),
            len(env_ids),
        )
        # self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
        #                                                 gymtorch.unwrap_tensor(self._effort_control),
        #                                                 gymtorch.unwrap_tensor(multi_env_ids_int32),
        #                                                 len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(indexes),
            len(env_ids),
        )

        ##################################################################
        # Reset mug (放在 table stand 上)
        ##################################################################
        sample_mug_pos = torch.zeros(num_resets, 3, device=self.device)
        pos_noise = torch.rand((num_resets, 3), device=self.device)

        # Sampling xy is "centered" around middle of table stand
        centered_table_stand_xy_state = torch.tensor(
            self._table_stand_surface_pos[:2],
            device=self.device,
            dtype=torch.float32,
        )
        sample_mug_pos[:, :2] = centered_table_stand_xy_state.unsqueeze(
            0
        ) + self.start_position_noise * 2.0 * (pos_noise[:, :2] - 0.5)

        # Set z value to be on table stand surface (使用初始位置避免陷進去)
        sample_mug_pos[:, 2] = self._mug_start_pos[2]
        self.root_pos[env_ids, self.handles["mug"], :] = sample_mug_pos[
            :, :3
        ]

        sample_mug_state = torch.zeros(num_resets, 13, device=self.device)
        sample_mug_state[:, 6] = 1.0
        aa_rot = torch.zeros(num_resets, 3, device=self.device)
        aa_rot[:, 2] = (
            np.pi + 2.0
            * self.start_rotation_noise
            * (torch.rand(num_resets, device=self.device) - 0.5)
        )
        self.root_rot[env_ids, self.handles["mug"], :] = quat_mul(
            axisangle2quat(aa_rot), sample_mug_state[:, 3:7]
        )

        # Update root state for mug
        multi_env_ids_int32 = self._global_indices[env_ids, -1:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew' + key] = self.episode_sums[key][env_ids]
            self.episode_sums[key][env_ids] = 0.


    def pre_physics_step(self, actions):
        if torch.rand(1).item() < 0.01:  # 偶爾印一次，避免太多
            print("u_gripper sample:", self.actions[:5, 13].detach().cpu().numpy())
        
        # 1. 动作处理与克隆
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone().to(self.device)

        # --- 增加嚴格的抓取條件判斷 ---
        # 為了決定是否執行夾爪閉合，我們需要檢查末端執行器(eef)的位置和姿態
        eef_pos = self.states["eef_pos"]
        handle_pos = self.states["handle_target_pos"]
        eef_rot = self.states["eef_rot"]
        handle_rot = self.states["handle_target_rot"]

        # 距離檢查
        align_dist = torch.linalg.norm(eef_pos - handle_pos, ord=2, dim=-1)

        # 姿態對齊檢查 (與獎勵函數中的邏輯相同)
        # 定義夾爪和把手的本地座標軸
        gripper_forward_axis = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        gripper_up_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        handle_inward_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        handle_up_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)

        # 將本地座標軸轉換到世界座標系
        axis1 = tf_vector(eef_rot, gripper_forward_axis)
        axis2 = tf_vector(handle_rot, handle_inward_axis)
        axis3 = tf_vector(eef_rot, gripper_up_axis)
        axis4 = tf_vector(handle_rot, handle_up_axis)

        # 計算點積
        dot1 = torch.bmm(axis1.view(self.num_envs, 1, 3), axis2.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)
        dot2 = torch.bmm(axis3.view(self.num_envs, 1, 3), axis4.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)

        # 定義嚴格的對齊條件
        # 夾爪的Z軸 (向前) 應與把手的X軸 (向內) 反向平行 (dot1 接近 -1)
        # 夾爪的X軸 (向上) 應與把手的Z軸 (向上) 平行 (dot2 接近 1)
        is_oriented = (dot1 < -0.9) & (dot2 > 0.9)
        is_close = (align_dist <= 0.025)  # 使用更小的閾值模擬 "觸碰"

        # 最終的抓取準備條件
        is_ready_to_grasp = is_close & is_oriented
        
        # 2. 初始化所有 DOF 目標為預設值
        self.diablo_dof_targets[:] = self.diablo_default_dof_pos.clone()
        
        # 3. 拆分动作指令
        # actions 格式: [右手關節動作(13個), gripper控制(1個)]
        # 右手關節對應 DOF 索引 15-27
        u_right_arm = self.actions[:, :13]  # 前13個是右手關節動作
        u_gripper = self.actions[:, 13:14]  # 最後1個是gripper控制

        # 4. 將右手關節動作映射到對應的 DOF 索引（15-27）
        # 將動作縮放並加上預設姿勢
        right_arm_targets = self.action_scale * u_right_arm + self.diablo_default_dof_pos[15:28]
        self.diablo_dof_targets[:, 15:28] = right_arm_targets

        # 5. 固定頭部和左手關節到預設值（確保不會動）
        # 頭部關節（索引 0-2）固定為 0
        self.diablo_dof_targets[:, 0:3] = 0.0
        # 左手關節（索引 3-14）固定到預設值
        self.diablo_dof_targets[:, 3:15] = self.diablo_default_dof_pos[3:15] 

        # 6. 夹爪（三指九关节）开关量控制逻辑
        # 如果 u_gripper >= 0 -> 闭合（上限）；否则 -> 张开（下限）
        self.gripper_lower_limits = self.diablo_dof_lower_limits[-9:]
        self.gripper_upper_limits = self.diablo_dof_upper_limits[-9:]
        
        # 核心邏輯修改：
        # 只有在滿足嚴格的「抓取準備條件」且「動作指令要求閉合」時才閉合
        # 否則一律保持張開
        should_close = is_ready_to_grasp & (u_gripper.view(-1) >= 0.0)


        # 根据单一的 u_gripper 值，对所有 9 个手指关节进行二值化切换
        targets_fingers = torch.where(
            should_close.unsqueeze(1).expand(-1, 9),
            self.gripper_upper_limits.expand(self.num_envs, 9),
            self.gripper_lower_limits.expand(self.num_envs, 9)
        )
        
        # 更新目标张量中的最后 9 个槽位（右侧夹爪手指）

        self.diablo_dof_targets[:, -9:] = targets_fingers

        # 7. 安全限幅 (Clamp) 与部署
        # 确保所有目标值都在硬件物理限制范围内
        self.diablo_dof_targets[:] = tensor_clamp(
            self.diablo_dof_targets, 
            self.diablo_dof_lower_limits, 
            self.diablo_dof_upper_limits
        )

        # 为了效率，只向仿真器发送一次指令
        self.gym.set_dof_position_target_tensor(
            self.sim, 
            gymtorch.unwrap_tensor(self.diablo_dof_targets)
        )   

    def post_physics_step(self):
        
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()


#####################################################################
###=========================jit functions=========================###
#####################################################################


# @torch.jit.script
def compute_diablo_reward(
    reset_buf, progress_buf, actions,
    eef_pos, handle_pos, eef_rot, handle_rot,
    finger_tip1_pos, finger_tip2_pos, finger_tip3_pos,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, lift_reward_scale, orientation_reward_scale,
    max_episode_length,
    object_pos, object_rot, initial_object_z
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # --- Stage 1: Reaching and Alignment Rewards ---

    # Distance from hand to the handle. This is the primary reward for approaching the target.
    d = torch.norm(eef_pos - handle_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward  # Square it to make it steeper
    # Bonus for being very close
    dist_reward = torch.where(d <= 0.05, dist_reward * 2.0, dist_reward)

    # Reward for aligning the gripper's orientation with the handle's orientation.
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

    # --- Stage 2: Grasping Attempt Reward ---

    # This is a new, crucial reward to encourage the agent to try closing the gripper.
    # We reward the agent if it's close to the handle AND commands the gripper to close.
    # The 'actions' tensor contains the gripper action at index 13.
    gripper_close_action = (actions[:, 13] >= 0.0)

    # Stricter grasping condition: check for orientation alignment in addition to distance.
    # - Gripper's Z-axis (forward) must align with handle's X-axis (inward). (dot1)
    # - Gripper's X-axis (up) must align with handle's Z-axis (up). (dot2)
    # dot1 should be close to -1 (anti-parallel) and dot2 close to 1 (parallel).
    is_oriented_for_grasp = (dot1 < -0.9) & (dot2 > 0.9)
    is_close_to_grasp = (d < 0.025) & is_oriented_for_grasp  # 4cm distance threshold + orientation alignment
    
    # Give a reward for attempting to grasp at the right moment.
    grasp_attempt_reward = torch.where(is_close_to_grasp & gripper_close_action, torch.ones_like(rot_reward) * 0.5, torch.zeros_like(rot_reward))

    # --- Stage 3: Lifting Reward (Simplified) ---

    # This reward is now much simpler. We reward any upward movement of the mug
    # as long as the agent is attempting to grasp it.
    object_height = object_pos[:, 2] - initial_object_z
    
    # Condition for lifting: gripper is trying to close and is near the handle.
    is_grasping = is_close_to_grasp & gripper_close_action

    # Reward for lifting, now only conditioned on a grasp attempt, not a perfect finger wrap.
    lift_reward = torch.zeros_like(rot_reward)
    lift_reward = torch.where(is_grasping & (object_height > 0.01), lift_reward + 1.0, lift_reward) # Initial lift
    lift_reward = torch.where(is_grasping & (object_height > 0.05), lift_reward + 1.5, lift_reward) # Higher lift

    # --- Stage 4: Orientation Reward (for keeping mug upright) ---

    # Define the mug's local "up" vector (assuming Z is up)
    mug_up_vec = torch.tensor([0.0, 0.0, 1.0], device=eef_pos.device).repeat(num_envs, 1)
    # Get the mug's current "up" vector in world coordinates
    world_mug_up = tf_vector(object_rot, mug_up_vec)
    # Define the world's "up" vector
    world_up = torch.tensor([0.0, 0.0, 1.0], device=eef_pos.device).repeat(num_envs, 1)
    # Calculate the dot product. A value of 1 means perfectly upright.
    dot_product = torch.bmm(world_mug_up.view(num_envs, 1, 3), world_up.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    # The reward is the dot product itself. We can square it to make the penalty for tilting steeper.
    orientation_reward = dot_product * dot_product
    # Only apply this reward when the mug is being lifted.
    lift_condition = is_grasping & (object_height > 0.01)
    orientation_reward = torch.where(lift_condition, orientation_reward, torch.zeros_like(orientation_reward))

    # --- Penalties ---

    # Regularization on the actions to prevent jerky movements.
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # --- Total Reward ---
    # The new grasp_attempt_reward is added here.
    # Note: The original finger_dist_reward and around_handle_reward can be removed or kept.
    # For simplicity and to test the new structure, we'll comment them out for now.
    rewards = dist_reward_scale * dist_reward \
        + rot_reward_scale * rot_reward \
        + grasp_attempt_reward \
        + lift_reward_scale * lift_reward \
        + orientation_reward_scale * orientation_reward \
        - action_penalty_scale * action_penalty
        # + around_handle_reward_scale * around_handle_reward \  # Temporarily disabled
        # + finger_dist_reward_scale * finger_dist_reward \      # Temporarily disabled

    # --- Resets ---
    # Reset if mug is lifted high enough or max length reached.
    reset_buf = torch.where(object_height > 0.1, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    # Reset if mug falls on the floor
    reset_buf = torch.where(object_pos[:, 2] < 0.1, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf

# @torch.jit.script
def tf_vector(rot, vec):
    # type: (Tensor, Tensor) -> Tensor
    return quat_apply(rot, vec)

# @torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos,
                             drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
                             ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    global_franka_rot, global_franka_pos = tf_combine(
        hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos)
    global_drawer_rot, global_drawer_pos = tf_combine(
        drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos)

    return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos