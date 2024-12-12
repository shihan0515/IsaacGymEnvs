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
        self.cfg["env"]["numObservations"] = 92

        if self.control_type == "joint":
            # actions include: joint (4) + ??bool gripper (1)
            self.cfg["env"]["numActions"] = 28  # TODO: Add gripper control
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
        self._std_dist = self.cfg["env"]["rewards"]["stdDist"]
        self._weight_dist = self.cfg["env"]["rewards"]["weightDist"]
        self._weight_dist_tanh = self.cfg["env"]["rewards"]["weightDistTanH"]
        self._weight_ori = self.cfg["env"]["rewards"]["weightOri"]
        self._weight_action_rate = self.cfg["env"]["rewards"]["weightActionRate"]
        self._weight_joint_vel = self.cfg["env"]["rewards"]["weightJointVel"]
        self._weightGripper = self.cfg["env"]["rewards"]["weightGripper"]
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
        diablo_asset_file = self.cfg["env"]["asset"]["assetFileNamediablo"]

        # diablo asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        diablo_asset = self.gym.load_asset(
            self.sim, asset_root, diablo_asset_file, asset_options
        )

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

        # cube asset
        asset_options = gymapi.AssetOptions()
        self._cube_size = self.cfg["env"]["asset"]["cubeSize"]
        cube_asset = self.gym.create_box(
            self.sim, *([self._cube_size] * 3), asset_options
        )
 
        # target asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        self._target_radius = self.cfg["env"]["asset"]["targetRadius"]
        target_asset = self.gym.create_sphere(
            self.sim, self._target_radius, asset_options
        )

        return {
            "diablo": diablo_asset,
            "table": table_asset,
            "table_stand": table_stand_asset,
            "cube": cube_asset,
            "target": target_asset,
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
            [400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 400, 400, 400, 400, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6],
            dtype=torch.float32,
            device=self.device,
        )
        diablo_dof_damping = torch.tensor(
            [80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 80, 80, 80, 80, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2 ],
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



        # start pose for cube
        cube_start_pose = gymapi.Transform()
        self._cube_start_pos = self._table_stand_surface_pos + np.array(
            [0.0, 0.0, self._cube_size / 2]
        )
        cube_start_pose.p = gymapi.Vec3(*self._cube_start_pos)
        cube_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # start pose for target
        target_start_pose = gymapi.Transform()
        self._target_start_pos = self._table_surface_pos + np.array(
            [10.0, 10.0, self._table_stand_height+ self._target_radius]
        )
        target_start_pose.p = gymapi.Vec3(*self._target_start_pos)
        target_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        return {
            "diablo": diablo_start_pose,
            "table": table_start_pose,
            "table_stand": table_stand_start_pose,
            "cube": cube_start_pose,
            "target": target_start_pose,
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
        cube_asset = assets_dict["cube"]
        target_asset = assets_dict["target"]

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
        cube_start_pose = start_poses_dict["cube"]
        target_start_pose = start_poses_dict["target"]

        # compute aggregate size
        num_diablo_bodies = self.gym.get_asset_rigid_body_count(diablo_asset)
        num_diablo_shapes = self.gym.get_asset_rigid_shape_count(diablo_asset)
        max_agg_bodies = (
            num_diablo_bodies + 4
        )  # 1 for table, 1 table stand, 1 cube, 1 target
        max_agg_shapes = (
            num_diablo_shapes + 4
        )  # 1 for table, 1 table stand, 1 cube, 1 target

        self.envs = []
        self.diablos = []
        self.cubes = []
        self.targets = []

        indexes_sim_diablo = []
        indexes_sim_cube = []
        indexes_sim_target = []

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
                env_ptr, diablo_asset, diablo_start_pose, "diablo", i, 0, 0
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

            # Create cube
            cube_actor = self.gym.create_actor(
                env_ptr, cube_asset, cube_start_pose, "cube", i, 2, 0
            )
            self.gym.set_rigid_body_color(
                env_ptr,
                cube_actor,
                0,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(0.0, 0.4, 0.1),
            )
            indexes_sim_cube.append(
                self.gym.get_actor_index(
                    env_ptr, cube_actor, gymapi.DOMAIN_SIM
                )
            )

            # Create target
            target_actor = self.gym.create_actor(
                env_ptr, target_asset, target_start_pose, "target", i, 3, 0
            )
            self.gym.set_rigid_body_color(
                env_ptr,
                target_actor,
                0,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(1.0, 0.0, 0.0),
            )
            indexes_sim_target.append(
                self.gym.get_actor_index(
                    env_ptr, target_actor, gymapi.DOMAIN_SIM
                )
            )

            self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.diablos.append(diablo_actor)
            self.cubes.append(cube_actor)
            self.targets.append(target_actor)

        self.indexes_sim_diablo = torch.tensor(
            indexes_sim_diablo, dtype=torch.int32, device=self.device
        )
        self.indexes_sim_cube = torch.tensor(
            indexes_sim_cube, dtype=torch.int32, device=self.device
        )
        self.indexes_sim_target = torch.tensor(
            indexes_sim_target, dtype=torch.int32, device=self.device
        )

        # Setup data
        self.init_data()


    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        diablo_handle = self.diablos[0]

        self.handles = {
            # # Franka
            # "hand": self.gym.find_actor_rigid_body_handle(
            #     env_ptr, diablo_handle, "panda_hand"
            # ),
            "plam": self.gym.find_actor_rigid_body_handle(
                env_ptr, diablo_handle, "r_wrist_Link"
            ),            
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
            # Cube
            "cube": self.gym.find_actor_handle(env_ptr, "cube"),
            # Target
            "target": self.gym.find_actor_handle(env_ptr, "target"),
        }



        # Right arm defaults

        # self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        self.diablo_default_dof_pos = torch.tensor(
            np.radians([0, 0, 0, 57, 80, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0, 0, -57, -80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
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
                # "eef_lf_pos": self.rigid_body_pos[
                #     :, self.handles["leftfinger_tip"]
                # ],
                # "eef_rf_pos": self.rigid_body_pos[
                #     :, self.handles["rightfinger_tip"]
                # ],
                # Cube
                "cube_pos": self.root_pos[:, self.handles["cube"]],
                "cube_rot": self.root_rot[:, self.handles["cube"]],
                # Target
                "target_pos": self.root_pos[:, self.handles["target"]],
                "target_rot": self.root_rot[:, self.handles["target"]],
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
            "cube_pos": gymutil.WireframeSphereGeometry(
                0.04,
                8,
                8,
                None,
                color=(0, 0, 1),
            ),
            "target_pos": gymutil.WireframeSphereGeometry(
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
        self.rew_buf[:], self.reset_buf[:], self.log_reward_tensor,self.gripper_upper_limits = compute_diablo_reward(
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length,
            self.states["eef_pos"],
            self.states["r_index_tip_pos"],
            self.states["r_mid_tip_pos"],
            self.states["r_thumb_tip_pos"],
            self.states["target_pos"],
            self.states["eef_rot"],
            self.states["target_rot"],
            self.actions,
            self.prev_actions,
            self.states["dof_vel"],
            self._std_dist,
            self._weight_dist,
            self._weight_dist_tanh,
            self._weight_ori,
            self._weight_action_rate,
            self._weight_joint_vel,
            self._weightGripper ,
            self.log_reward_tensor,
            self.diablo_dof_lower_limits,
            self.diablo_dof_upper_limits,
            self.gripper_upper_limits,
            self.u_fingers,

            
            )
        self.episode_sums["pos_err_penalty_"] += self.log_reward_tensor[0]
        self.episode_sums["pos_err_tanh_"] += self.log_reward_tensor[1]
        self.episode_sums["action_rate_penalty_"] += self.log_reward_tensor[2]
        self.episode_sums["joint_vel_penalty_"] += self.log_reward_tensor[3]

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
        self.obs_buf[:, 57:60] = self.states["target_pos"]
        self.obs_buf[:, 60:64] = self.states["target_rot"]
        
        # need to be modified
        self.obs_buf[:, 64:92] = self.actions

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
        # Reset cube
        ##################################################################
        sample_cube_state = torch.zeros(num_resets, 13, device=self.device)
        # initialize rotation, quat w = 1.0
        sample_cube_state[:, 6] = 1.0
        pos_noise = torch.rand((num_resets, 3), device=self.device)

        # Sampling xy is "centered" around middle of table
        centered_table_xy_state = torch.tensor(
            self._table_surface_pos[:2],
            device=self.device,
            dtype=torch.float32,
        )
          # Sampling xy is "centered" around middle of table
        centered_table_stand_xy_state = torch.tensor(
            self._table_stand_surface_pos[:2],
            device=self.device,
            dtype=torch.float32,
        )
        sample_cube_state[:, :2] = centered_table_xy_state.unsqueeze(
            0
        ) + self.start_position_noise * 2.0 * (pos_noise[:, :2] - 0.5)

        # set z value, which is fixed height
        sample_cube_state[:, 2] = self._cube_start_pos[2]
        self.root_pos[env_ids, self.handles["cube"], :] = sample_cube_state[
            :, :3
        ]

        aa_rot = torch.zeros(num_resets, 3, device=self.device)
        aa_rot[:, 2] = (
            2.0
            * self.start_rotation_noise
            * (torch.rand(num_resets, device=self.device) - 0.5)
        )
        self.root_rot[env_ids, self.handles["cube"], :] = quat_mul(
            axisangle2quat(aa_rot), sample_cube_state[:, 3:7]
        )

        ##################################################################
        # Reset target
        ##################################################################
        sample_target_pos = torch.zeros(num_resets, 3, device=self.device)
        pos_noise = torch.rand((num_resets, 3), device=self.device)

        # Sampling xy is "centered" around middle of table
        sample_target_pos[:, :2] = centered_table_stand_xy_state.unsqueeze(
            0
        ) + self.start_position_noise * 2.0 * (pos_noise[:, :2] - 0.5)

        # Set z value, minimum is the start_position_noise
        sample_target_pos[:, 2] = (
            self._target_start_pos[2]
            + self.start_position_noise * 2.0 * pos_noise[:, 2]
        )
        self.root_pos[env_ids, self.handles["target"], :] = sample_target_pos[
            :, :3
        ]

        sample_target_state = torch.zeros(num_resets, 13, device=self.device)
        sample_target_state[:, 6] = 1.0
        aa_rot = torch.zeros(num_resets, 3, device=self.device)
        aa_rot[:, 2] = (
            2.0
            * self.start_rotation_noise
            * (torch.rand(num_resets, device=self.device) - 0.5)
        )
        aa_rot[:, 0] = np.pi
        self.root_rot[env_ids, self.handles["target"], :] = quat_mul(
            axisangle2quat(aa_rot), sample_target_state[:, 3:7]
        )

        # Update root state for cube and target
        multi_env_ids_int32 = self._global_indices[env_ids, -2:].flatten()
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
        #check actions has nan
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone().to(self.device)
        # assert not torch.isnan(actions).any(), "Actions has nan"
        # if torch.isnan(actions).any():
        #     print(f"Actions contain NaN, replacing with 0 at indices: {torch.isnan(actions).nonzero()}")
        #     actions[torch.isnan(actions)] = 0.0



        # TODO: Implement gripper control from actions (currently not used)
        # Split arm and gripper command
        # u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        self.actions[..., 0:2] = 0.0  # Remove head action
        self.actions[..., [6,7,8,9,10,11,12,13,14]] = 0.0 # Remove left griper action
        # self.actions[..., [19,20,21,22,23,24,25,26,27]] = 0


        targets = self.action_scale * self.actions + self.diablo_default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))
        
    

        u_arm, u_gripper = self.actions[:, :-9], self.actions[:, -9: ]
        u_fingers = torch.zeros((self.num_envs, 9), dtype=torch.float32, device=self.device)
        # for i in range(9):
        #     u_fingers[:, i] = u_gripper[:, i]
        #     # self.diablo_dof_lower_limits[-9 + i].item()
        #     self.diablo_dof_upper_limits[-9 + i].item()
        # Update u_fingers with u_gripper values for the last 9 DOFs
        u_fingers[:, :9] = u_gripper

        # Access the last 9 upper limits directly as a tensor or list
        self.gripper_upper_limits = self.diablo_dof_upper_limits[-9:]
        print("gripper_upper_limits",self.gripper_upper_limits)



        if self.control_type == "joint": 
            targets_arm =(
                self.diablo_dof_targets[:, :-9]
                + self.diablo_dof_speed_scales[:-9] 
                * self.dt
                * u_arm
                * self._action_scale 
            )   

        # 夾緊並設置目標位置
        self.diablo_dof_targets = tensor_clamp(
            targets, self.diablo_dof_lower_limits, self.diablo_dof_upper_limits
        )


        # Deploy actions
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.diablo_dof_targets)
        )
        
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

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


@torch.jit.script
def compute_diablo_reward(
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float,
    end_effector_pos: torch.Tensor,
    d_r_index_tip_pos: torch.Tensor,
    d_r_mid_tip_pos: torch.Tensor,
    d_r_thumb_tip_pos: torch.Tensor,
    target_pos: torch.Tensor,
    end_effector_ori: torch.Tensor,
    target_ori: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    dof_vel: torch.Tensor,
    std_dist: float,
    weight_dist: float,
    weight_dist_tanh: float,
    weight_ori: float,
    weight_action_rate: float,
    weight_joint_vel: float,
    weightGripper: float,
    log_reward_tensor: torch.Tensor,
    diablo_dof_lower_limits: torch.Tensor,
    diablo_dof_upper_limits: torch.Tensor,
    gripper_upper_limits: torch.Tensor,
    u_fingers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor,torch.Tensor,torch.Tensor]:
    # Position reward
    position_err = torch.norm(end_effector_pos - target_pos, dim=-1)
    position_err_tanh = 1 - torch.tanh(position_err * std_dist)
    # d_r_index_tip_pos_err = torch.norm(d_r_index_tip_pos - target_pos, dim=-1)
    # d_r_mid_tip_pos_err = torch.norm(d_r_mid_tip_pos - target_pos, dim=-1)
    # d_r_thumb_tip_pos_err = torch.norm(d_r_thumb_tip_pos - target_pos, dim=-1)
    # dist_reward = 1 - torch.tanh(10.0 * (d_r_index_tip_pos_err + d_r_mid_tip_pos_err + d_r_thumb_tip_pos_err + position_err) / 4)
    gripped = torch.where(position_err[:, None] <= 0.1 ,gripper_upper_limits ,diablo_dof_lower_limits[-9:][None, :])
    # Orientataion reward
    # orientation_err = quat_diff_rad(end_effector_ori, target_ori)
    print("gripped", gripped)
    # print("grippedshape",gripped.shape)
    # # print("gr"gripped.shape[0])
    # print("position_err[:, None]",position_err[:, None])
    # Action penalty
    action_rate = torch.sum(torch.square(actions - prev_actions), dim=1)
    joint_vel = torch.sum(torch.square(dof_vel), dim=1)

    total_position_err = position_err * (-(weight_dist**2)) 
    total_position_err_tanh = position_err_tanh * weight_dist_tanh
    total_action_rate = action_rate * weight_action_rate
    total_joint_vel = joint_vel * weight_joint_vel
    # print(position_err)
    # gripped = dof_states[:, -9:]  # 最後九個關節的當前角度
    # target_grip_condition = torch.all(gripped >= gripper_upper_limits * 0.95, dim=1)

    # 設置新的 reward 項目
    # grip_reward = torch.zeros_like(position_err)  # 初始化
    # grip_reward[target_grip_condition] = 1.0     # 符合條件時給予獎勵
    # if torch.all(position_err <= 0.001) and torch.all(gripped == gripper_upper_limits):
    #     success_gripper = 1
    # elif torch.any(position_err > 0.001) and torch.all(gripped == diablo_dof_lower_limits[-9:]):
    #     success_gripper = 0
    # if torch.all(position_err <= 0.001) and torch.all(gripped == gripper_upper_limits):
    #     success_gripper = -1
    # success_gripper = torch.zeros(position_err.shape[0], device=position_err.device)
    # print("position_err.shape",position_err.shape)  # Ensure it has at least 2 dimensions.
    # print("gripped.shape",gripped.shape)       # Ensure it has at least 2 dimensions.
    # print("diablo_dof_lower_limits.shape",diablo_dof_lower_limits[-9:].shape)  # Should align with `gripped`.
    # print("position_err",position_err)
    # Initialize success_gripper
    success_gripper = torch.zeros(position_err.shape[0], device=position_err.device, dtype=torch.float)
    # print("position_err: ", position_err)
    # Apply success condition
    success_gripper[
        (gripped == gripper_upper_limits).all(dim=1)
    ] = 1.0

    # Apply failure condition
    success_gripper[
        (position_err > 0.01) | (gripped == diablo_dof_lower_limits[-9:]).all(dim=1)
    ] = -1.0
    # Initialize success_gripper
    # success_gripper = torch.zeros(position_err.shape[ 0 ], device=position_err.device, dtype= torch.float )

    print("success_gripper: ", success_gripper)

    reach_goal = (position_err <= 0.001) 
    rewards = torch.zeros_like(position_err)

    # Compose rewards
    rewards += (
        # (~reach_goal).float() * (position_err * weight_dist + position_err_tanh * weight_dist_tanh)
        position_err * (-(weight_dist**2)) 
        +position_err_tanh * weight_dist_tanh

        # + orientation_err * weight_ori
        # +dist_reward * 0.2

        + action_rate * weight_action_rate
        + joint_vel * weight_joint_vel
        + success_gripper * weightGripper
    )

    # rewards +=((target_reached & grip_condition) * reward_grip_success)

    # print("gripper_upper_limits",gripper_upper_limits)
    # print("gripped",gripped)

    log_reward_tensor[0] = total_position_err
    log_reward_tensor[1] = total_position_err_tanh
    log_reward_tensor[2] = total_action_rate
    log_reward_tensor[3] = total_joint_vel

    # log_reward_tensor[0] = pos
    # ition_err
    # print("success_gripper: ", success_gripper)
    # print("position_err * weight_dist: ", total_position_err)
    # # print("position_err_tanh * weight_dist_tanh: ", position_err_tanh * weight_dist_tanh)
    # print("action_rate * weight_action_rate: ", total_action_rate)
    # print("joint_vel * weight_joint_vel: ", total_joint_vel)
    # # Compute resets
 



    reset_buf = torch.where(
        reach_goal, torch.ones_like(reset_buf), reset_buf
    )

    reset_buf = torch.where(
        (progress_buf >= max_episode_length - 1)
        ,torch.ones_like(reset_buf),
        reset_buf,
    )


    return rewards, reset_buf, log_reward_tensor,gripper_upper_limits
# u_fingers

