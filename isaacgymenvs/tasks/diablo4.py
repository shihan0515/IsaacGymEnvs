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
from typing import Dict
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils import isaacgym_utils
from isaacgymenvs.utils.torch_jit_utils import quat_to_euler, to_torch, get_axis_params, torch_rand_float, quat_rotate, quat_rotate_inverse, get_euler_xyz
# from pynput import keyboard
from isaacgymenvs.utils.torch_jit_utils import *


class diablo22(VecTask):
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
        self.cfg["env"]["numObservations"] = 121

        if self.control_type == "joint":
            # actions include: joint (4) + ??bool gripper (1)
            self.cfg["env"]["numActions"] = 34  # TODO: Add gripper control
        elif self.control_type == "cartesian":
            self.cfg["env"]["numActions"] = 3

        self.actions = torch.zeros(
            (self.cfg["env"]["numEnvs"], self.cfg["env"]["numActions"]),
            dtype=torch.float32,
            device=sim_device,
        )
        self.prev_actions = torch.zeros_like(self.actions)


        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]
        self.wheel_action_scale = self.cfg["env"]["control"]["wheelActionScale"]

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
        
        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["height"] = self.cfg["env"]["learn"]["heightRewardScale"]
        self.rew_scales["pitch"] = self.cfg["env"]["learn"]["pitchRewardScale"]
        self.rew_scales["sync"] = self.cfg["env"]["learn"]["syncRewardScale"]
        self.rew_scales["fall"] = self.cfg["env"]["learn"]["fallRewardScale"]
        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]     

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        self.up_axis = "z"
        self.up_axis_idx = 2
        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )
        
        # # self.dt = self.sim_params.dt
        # # other
        # self.dt: float = self.sim_params.dt

        self.dt = self.sim_params.dt   
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        # self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)           
        
        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt       
       
        # Values to be filled in at runtime
        self.states = {}

        self.up_axis = "z"
        self.up_axis_idx = 2

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
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_assets(self):
        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.cfg["env"]["asset"]["assetRoot"],
        )
        diablo_asset_file = self.cfg["env"]["asset"]["assetFileNamediablo"]

        # diablo asset
        diablo_asset_options = gymapi.AssetOptions()
        diablo_asset_options.flip_visual_attachments = False
        diablo_asset_options.fix_base_link = False
        diablo_asset_options.collapse_fixed_joints = False
        diablo_asset_options.replace_cylinder_with_capsule = True

        diablo_asset_options.disable_gravity = False
        diablo_asset_options.angular_damping = 0.0
        diablo_asset_options.linear_damping = 0.0
        diablo_asset_options.armature = 0.0        
        diablo_asset_options.thickness = 0.001
        diablo_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        diablo_asset_options.use_mesh_materials = True
        diablo_asset = self.gym.load_asset(
            self.sim, asset_root, diablo_asset_file, diablo_asset_options
        )

        # # table asset
        self._table_thickness = self.cfg["env"]["asset"]["tableThickness"]
        # asset_options = gymapi.AssetOptions()
        # asset_options.fix_base_link = True
        # table_asset = self.gym.create_box(
        #     self.sim, *[0.5, 0.5, self._table_thickness], asset_options
        # )

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
            # "table": table_asset,
            "diablo_asset_options": diablo_asset_options, 
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
        # print("diablo rigid body dict: ", self.rigid_body_dict_diablo)
        print("num diablo bodies: ", self.num_diablo_bodies)
        print("num diablo dofs: ", self.num_diablo_dofs)
        asset_options = self._create_assets()
        # asset_options = gymapi.AssetOptions()
        # asset_options.collapse_fixed_joints = True   
        # 獲取 DOF 名稱並打印
        self.dof_names = self.gym.get_asset_dof_names(diablo_asset)
        # print("All DOF names: ", self.dof_names)
        body_names = self.gym.get_asset_rigid_body_names(diablo_asset)

        # print("All body names: ", body_names)
        dof_props = self.gym.get_asset_dof_properties(diablo_asset)
        dof_dict = self.gym.get_asset_dof_dict(diablo_asset)
        print("dof dict: ", dof_dict)
        self.shoPitchIdx = []
        self.shoRollIdx = []
        self.wheelIdx = []
        self.kneeIdx = []
        self.hipIdx = []
        self.elIdx = []              
        self.diablo_dof_lower_limits = []
        self.diablo_dof_upper_limits = []

        for name,idx in dof_dict.items():
            dof_props['driveMode'][idx] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][idx] = self.cfg["env"]["control"]["stiffness"][name]
            dof_props['damping'][idx] = self.cfg["env"]["control"]["damping"][name]

            if name == 'l4' or name == 'r4':
                dof_props['driveMode'][idx] = gymapi.DOF_MODE_VEL
                self.wheelIdx.append(idx)

            elif name == 'l3' or name == 'r3':
                self.kneeIdx.append(idx)

            elif name == 'l1' or name == 'r1':
                self.hipIdx.append(idx)

            elif name == 'r_sho_roll' or name == 'l_sho_roll':
                self.shoRollIdx.append(idx)

            elif name == 'r_el' or name == 'l_el':
                self.elIdx.append(idx)
            
            elif name == 'r_sho_pitch' or name == 'l_sho_pitch':
                self.shoPitchIdx.append(idx)

            self.diablo_dof_lower_limits = torch.tensor(diablo_dof_props["lower"], device=self.device)
            self.diablo_dof_upper_limits = torch.tensor(diablo_dof_props["upper"], device=self.device)


        self.diablo_dof_lower_limits = to_torch(
            self.diablo_dof_lower_limits, device=self.device
        )
        self.diablo_dof_upper_limits = to_torch(    
            self.diablo_dof_upper_limits, device=self.device
        )
        self.diablo_dof_speed_scales = torch.ones_like(
            self.diablo_dof_lower_limits
        )
        # print("dof_pos:", diablo.dof_pos)
        # print("dof_lower:", self.diablo_dof_upper_limits)
        # print("dof_upper:", self.diablo_dof_lower_limits)


        return diablo_dof_props
    
  

    def _init_start_poses(self):
        # start pose for franka
        diablo_start_pose = gymapi.Transform()
        diablo_start_pose.p = gymapi.Vec3(
            -0.17,
            0.0,
            0.45,
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
            0.2,
            0.0,
            0.35,
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
            [10.0, 10.0, self._table_stand_height+ self._target_radius+ 0.05]
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
        diablo_asset_options = assets_dict["diablo_asset_options"]
        diablo_asset = assets_dict["diablo"]
        # table_asset = assets_dict["table"]
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
            num_diablo_bodies + 3
        )  # 1 for table, 1 table stand, 1 cube, 1 target
        max_agg_shapes = (
            num_diablo_shapes + 3
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

            # Create diablo
            diablo_actor = self.gym.create_actor(
                env_ptr, diablo_asset, diablo_start_pose, "diablo", i, 0, 0
            )
            self.gym.set_actor_dof_properties(
                env_ptr, diablo_actor, diablo_dof_props
            )
            self.gym.enable_actor_dof_force_sensors(env_ptr, diablo_actor)

            indexes_sim_diablo.append(
                self.gym.get_actor_index(
                    env_ptr, diablo_actor, gymapi.DOMAIN_SIM
                )
            )

            # Create table
            # self.gym.create_actor(
            #     env_ptr, table_asset, table_start_pose, "table", i, 1, 0
            # )
            self.gym.create_actor(
                env_ptr,
                table_stand_asset,
                table_stand_start_pose,
                "table_stand", i,1,0,
            )

            # Create cube
            cube_actor = self.gym.create_actor(
                env_ptr, cube_asset, cube_start_pose, "cube", i, 0, 0
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
                env_ptr, target_asset, target_start_pose, "target", i, 0, 0
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

        # self.base_index = self.gym.find_actor_rigid_body_handle(
        #     self.envs[0], self.diablos[0], "lower_base_link")
        
        # asset_options = gymapi.AssetOptions()        
        # asset_options.collapse_fixed_joints = True
        # asset_options.replace_cylinder_with_capsule = True
        # asset_options.flip_visual_attachments = False
        # asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        # asset_options.disable_gravity = False
        # asset_options.angular_damping = 0.0
        # asset_options.linear_damping = 0.0
        # asset_options.armature = 0.0        
        # asset_options.thickness = 0.001
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # asset_options.use_mesh_materials = True
        # asset_options.collapse_fixed_joints = True
       
        # body_names = self.gym.get_asset_rigid_body_names(diablo_asset)
        self.dof_names = self.gym.get_asset_dof_names(diablo_asset)
        self.num_dof = self.gym.get_asset_dof_count(diablo_asset)

        # 建立 DOF 名稱對應索引
        self.dof_name_to_id = {k: v for k, v in zip(self.dof_names, np.arange(self.num_dof))}

        # 建立剛體與形狀對應表
        self.num_rgbd = self.gym.get_asset_rigid_body_count(diablo_asset)
        self.rgid_shape_to_id = {}
        self.rgid_body_to_id = {}
        self.rgid_body_to_name = {}
        # shape_indices = self.gym.get_asset_rigid_body_shape_indices(diablo_asset)
        # body_names = self.gym.get_asset_rigid_body_names(diablo_asset)  # List[str]

        # 把全部剛體名稱一次拿出來（這樣才可以篩選）
        body_names = self.gym.get_asset_rigid_body_names(diablo_asset)  # List[str]
        print("all body names: ", body_names)

        # 儲存每個剛體名稱的 index
        self.rgid_body_to_id = {name: i for i, name in enumerate(body_names)}
        self.rgid_body_to_name = {i: name for i, name in enumerate(body_names)}

        # shape_indices（先拿出來一次）
        shape_indices = self.gym.get_asset_rigid_body_shape_indices(diablo_asset)
        self.rgid_shape_to_id = {
            name: shape_indices[i].start
            for i, name in enumerate(body_names) if shape_indices[i].count > 0
        }

        # 搜尋腳部或輪子名稱
        extremity_name = "wheel"
        feet_names = [name for name in body_names if extremity_name in name]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)

        # # # 對應到 feet indices
        self.feet_indices = torch.tensor(
            [self.rgid_body_to_id[name] for name in feet_names],
            dtype=torch.long, device=self.device
        )        
        
        # for i in range(len(feet_names)):
        #     self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.diablos[0], feet_names[i])
      
        

        # 假設你已經有 diablo_asset
        num_bodies = self.gym.get_asset_rigid_body_count(diablo_asset)
        print("extrmity name: ", extremity_name)
        print('feet_names',feet_names)
        # 自動取得所有剛體名稱（URDF link 對應）
       
     
        exclude_names = {"wheel_right_link_1", "wheel_left_link_1"}
        knee_names = []
        # knee_names = ['base_link', 'head_pan_Link', 'head_tilt_Link', 'l_sho_pitch_Link',
        #                'l_sho_roll_Link', 'l_el_Link', 'l_wrist_Link', 'l_index_base_Link',
        #                  'l_index_middle_Link', 'l_index_tip_Link', 'l_mid_base_Link',
        #                    'l_mid_middle_Link', 'l_mid_tip_Link', 'l_thumb_base_Link',
        #                      'l_thumb_middle_Link', 'l_thumb_tip_Link','lower_base_link',
        #                        'motor_left_link_1', 'leg_left_link_1', 'leg2_left_link_1',
        #                           'motor_right_link_1', 'leg_right_link_1',
        #                            'leg2_right_link_1', 'r_sho_pitch_Link',
        #                              'r_sho_roll_Link', 'r_el_Link', 'r_wrist_Link','panda_grip_site','r_index_base_Link',
        #                                'r_index_middle_Link', 'r_index_tip_Link', 'r_mid_base_Link',
        #                                  'r_mid_middle_Link', 'r_mid_tip_Link', 'r_thumb_base_Link',
        #                                    'r_thumb_middle_Link', 'r_thumb_tip_Link']
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
       
       
        for i in range(num_bodies):
            name = self.gym.get_asset_rigid_body_name(diablo_asset, i)

            self.rgid_body_to_id[name] = i
            self.rgid_body_to_name[i] = name

            if name not in exclude_names:
                knee_names.append(name)

        # 儲存所有 knee 的 index（排除輪子）
        self.knee_indices = torch.tensor(
            [self.rgid_body_to_id[name] for name in knee_names],
            dtype=torch.long,
            device=self.device
        )

        # self.base_index = 0
        # self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        # for i in range(len(feet_names)):
        #     self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.diablos[0], feet_names[i])
        # for i in range(len(knee_names)):
        #     self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.diablos[0], knee_names[i])

        # for name in knee_names:
        #     print(f"{name}: {self.rgid_body_to_id[name]}")  

        self.base_index = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.diablos[0], "lower_base_link")
        print("base_index:", self.base_index)
        # self.base_index = torch.tensor([16], device=self.device, dtype=torch.long)
        print("knee_names:", knee_names)
        print("knee_indices1:", self.knee_indices)
        # Setup data
        self.init_data()


    def init_data(self):
        # Setup sim handles
        
        env_ptr = self.envs[0]
        diablo_handle = self.diablos[0]
        print("env_ptr:", env_ptr)
        print("diablo_handle:", diablo_handle)
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
            "lower_base_link": self.gym.find_actor_rigid_body_handle(
                env_ptr, diablo_handle, "lower_base_link"
            ),
          
            # Cube
            "cube": self.gym.find_actor_handle(env_ptr, "cube"),
            # Target
            "target": self.gym.find_actor_handle(env_ptr, "target"),
        }

        # self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        # self.base_index = self.gym.find_actor_rigid_body_handle(
        #     env_ptr, diablo_handle, "lower_base_link"
        # )
        # print("base_index:", self.base_index)
        # Right arm defaults



        # self.diablo_default_dof_pos = torch.tensor(
        #     np.radians([0, 0, 0, 57, 80, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0, 0, 0, 0, 0, 0, 0, 0, -57, -80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        #     device=self.device,
        #     dtype=torch.float32,
        # )
        # # Right arm defaults
        # self.diablo_default_dof_pos = torch.tensor(
        #     np.radians([0 , -57, -80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),s
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


        self.u_fingers = torch.zeros((self.num_envs, 9), dtype=torch.float32, device=self.device)




        # Setup tensor buffers and views: roots, DOFs, rigid bodies.
        root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(
            self.sim
        )
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

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
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_diablo_dofs)
        
        self.extras = {}
        self.initial_root_states = self.root_state.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
    
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        if self.control_type == "cartesian":
            self.jacobian = gymtorch.wrap_tensor(jacobian_tensor)
        ##
        self.actor_root_pos = self.initial_root_states[:, :3].view(self.num_envs, -1, 3)
        self.actor_root_rot = self.initial_root_states[:, 3:7].view(self.num_envs, -1, 4)
        self.actor_root_vel_lin = self.initial_root_states[:, 7:10].view(
            self.num_envs, -1, 3
        )
        self.actor_root_vel_ang = self.initial_root_states[:, 10:13].view(
            self.num_envs, -1, 3
        )
        # Root states
        self.root_pos = self.root_state[:, :3].view(self.num_envs, -1, 3)
        self.root_rot = self.root_state[:, 3:7].view(self.num_envs, -1, 4)
        self.root_vel_lin = self.root_state[:, 7:10].view(self.num_envs, -1, 3)
        self.root_vel_ang = self.root_state[:, 10:13].view(
            self.num_envs, -1, 3
        )

        # DoF states
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_diablo_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_diablo_dofs, 2)[..., 1]


        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        self.commands = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_x = self.commands.view(self.num_envs, 2)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 2)[..., 1]

        print("commands shape:", self.commands.shape)

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

                # "base_quat": self.root_rot[:, 0],           # (num_envs, 4)
                # "base_lin_vel": self.root_vel_lin[:, 0],    # (num_envs, 3)
                # "base_ang_vel": self.root_vel_ang[:, 0],    # (num_envs, 3)

                "base_quat": self.actor_root_rot[:, 0],           # (num_envs, 4)
                "base_lin_vel": self.actor_root_vel_lin[:, 0],    # (num_envs, 3)
                "base_ang_vel": self.actor_root_vel_ang[:, 0], 
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
        self.gym.refresh_dof_force_tensor(self.sim)
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

        # 1. 把全局 root_state 拆成 (num_envs, num_actors, 13)
        # 原本 self.root_state.shape = (num_envs * num_actors, 13)
        num_actors = self.root_state.shape[0] // self.num_envs
        all_root = self.root_state.view(self.num_envs, num_actors, 13)

        # 2. 取出 robot（actor idx = 0）
        root_robot = all_root[:, 0, :]                # shape → (num_envs, 13)
        #    如果 robot 不是第 0 個就改成 [:, your_robot_actor_idx, :]

        # 3. 同理拆 contact_forces（如果你 reward 裡有用到 contact_forces）
        #    原 contact_forces.shape = (num_envs * num_bodies, 3)
        num_bodies = self.contact_forces.shape[0] // self.num_envs
        cf = self.contact_forces.view(self.num_envs, -1, 3)

        self.rew_buf[:], self.reset_buf[:], self.log_reward_tensor,self.gripper_upper_limits = compute_diablo_reward(
            root_robot,
            self.commands,
            root_robot[:, 7:10],                       # base_lin_vel in world
            root_robot[:, 10:13],                      # base_ang_vel in world
            root_robot[:, 3:7],                        # base_quat
            self.torques,
            self.actions,
            cf,                                        # (num_envs, num_bodies, 3)
            self.knee_indices,
            self.progress_buf,
            self.rew_scales,
            self.lin_vel_scale,
            self.ang_vel_scale,
            self.base_index,
            self.max_episode_length,

            self.states["eef_pos"],
            self.states["target_pos"],
            self.prev_actions,
            self.states["dof_vel"],
            self._std_dist,
            self._weight_dist,
            self._weight_dist_tanh,

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
        EPSILON = 1e-6  # Small value to prevent division by zero
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # base info
        base_quat = self.states["base_quat"]  # (num_envs, 4)
        base_euler = quat_to_euler(base_quat)  # (num_envs, 3)
        base_pos = self.root_state[:, :3]

        base_lin_vel = self.states["base_lin_vel"] # (num_envs, 3)
        base_ang_vel = self.states["base_ang_vel"]  # (num_envs, 3)

        # dof
        dof_pos_scaled = (self.dof_pos - self.default_dof_pos) * self.dof_pos_scale  # (num_envs, dof)
        dof_vel_scaled = self.dof_vel * self.dof_vel_scale  # (num_envs, dof)

        # command
        commands_scaled = self.commands * torch.tensor(
            [self.lin_vel_scale, self.ang_vel_scale],
            device=self.device
        )  # (num_envs, 2)

        # ----- fill obs_buf -----
        self.obs_buf[:, 0] = self.progress_buf / self.max_episode_length  # (num_envs,)

        self.obs_buf[:, 1:35] = dof_pos_scaled  # (num_envs, 34)
        self.obs_buf[:, 35:69] = dof_vel_scaled[:, :34]  # (num_envs, 34)

        self.obs_buf[:, 69:72] = torch.nan_to_num(self.states["target_pos"], nan=0.0)  # (num_envs, 3)
        self.obs_buf[:, 72:76] = torch.nan_to_num(self.states["target_rot"], nan=0.0)  # (num_envs, 4)

        self.obs_buf[:, 76:110] = torch.nan_to_num(self.actions, nan=0.0)  # (num_envs, 34)

        self.obs_buf[env_ids, 110:113] = base_lin_vel[env_ids]
        self.obs_buf[env_ids, 113:116] = base_ang_vel[env_ids]
        self.obs_buf[env_ids, 116:119] = base_euler[env_ids]
        self.obs_buf[env_ids, 119:121] = commands_scaled[env_ids]


        # 可選的重力方向投影（例如觀察重力在 base frame 下的方向）
        # self.obs_buf[:, 121:124] = projected_gravity

    # def compute_observations(self):
    #     self._refresh()
    #     EPSILON = 1e-6  # Small value to prevent division by zero

    #     dof_range = self.diablo_dof_upper_limits - self.diablo_dof_lower_limits
    #     dof_pos_scaled = 2.0 * (self.states["dof_pos"] - self.diablo_dof_lower_limits) / (dof_range + EPSILON) - 1.0

    #     dof_pos_scaled = (
    #         2.0
    #         * (self.states["dof_pos"] - self.diablo_dof_lower_limits)
    #         / dof_range-1.0
            
    #     )
    #     dof_vel_scaled = self.states["dof_vel"] * self.dof_vel_scale
    #     dof_vel_scaled = torch.nan_to_num(self.states["dof_vel"], nan=0.0) * self.dof_vel_scale

    #     valid_range = (self.diablo_dof_upper_limits - self.diablo_dof_lower_limits) < 1e10  # Filter out extreme values
    #     dof_pos_scaled = (self.diablo_dof_upper_limits - self.diablo_dof_lower_limits) / torch.where(valid_range, self.diablo_dof_upper_limits - self.diablo_dof_lower_limits, torch.tensor(1.0, device='cuda:0'))

    #     self.obs_buf[:, 0] = self.progress_buf / self.max_episode_length
    #     # print(f"dof_pos_scaled shape: {dof_pos_scaled.shape}")
    #     self.obs_buf[:, 1:35] = dof_pos_scaled.unsqueeze(0)  # Adds batch dimension

    #     # self.obs_buf[:, 1:35] = dof_pos_scaled[:, :34]
    #     self.obs_buf[:, 35:69] = dof_vel_scaled[:, :34]
    #     # print("dof_pos:", self.states["dof_pos"])
    #     if torch.isnan(self.states["target_pos"]).any():
    #         print("WARNING: target_pos contains NaN")
    #     self.obs_buf[:, 69:72] = torch.nan_to_num(self.states["target_pos"], nan=0.0)
    #     # 檢查 target_rot
    #     if torch.isnan(self.states["target_rot"]).any():
    #         print("WARNING: target_rot contains NaN")
    #     self.obs_buf[:, 72:76] = torch.nan_to_num(self.states["target_rot"], nan=0.0)   


    #     if torch.isnan(self.actions).any():
    #         print("WARNING: actions contain NaN")
    #     self.obs_buf[:, 76:110] = torch.nan_to_num(self.actions, nan=0.0)

    #         # ✅ 檢查哪個部分是 NaN
    #     if torch.isnan(self.obs_buf).any():
    #         print("WARNING: obs_buf contains NaN!")
    #         print("NaN in progress_buf:", torch.isnan(self.progress_buf).any())
    #         print("NaN in dof_pos_scaled:", torch.isnan(dof_pos_scaled).any())
    #         print("NaN in dof_vel_scaled:", torch.isnan(dof_vel_scaled).any())
    #         print("NaN in target_pos:", torch.isnan(self.states["target_pos"]).any())
    #         print("NaN in target_rot:", torch.isnan(self.states["target_rot"]).any())
    #         print("NaN in actions:", torch.isnan(self.actions).any())  # 這是來自上一個 step 的動作

    def reset_idx(self, env_ids):

        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        ##################################################################
        # Reset diablo
        ##################################################################
        num_resets = len(env_ids)
        print("num_resets: ", num_resets)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_diablo_dofs), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_diablo_dofs), device=self.device)

        # dof_noise = torch.rand(
        #     (num_resets, self.num_diablo_dofs), device=self.device
        # )
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        # pos = self.diablo_default_dof_pos.unsqueeze(
        #     0
        # ) + self.diablo_dof_noise * 2.0 * (dof_noise - 0.5)
        # pos = tensor_clamp(
        #     pos, self.diablo_dof_lower_limits, self.diablo_dof_upper_limits
        # )
        # # print("env_ids: ", env_ids)
        # # Overwrite gripper init pos
        # # (no noise since these are always position controlled)
        # pos[:, -12:-3] = self.diablo_default_dof_pos[-12:-3]
        # # pos[:, -2:] = 0.0 # close gripper

        # self.diablo_dof_targets[env_ids, :] = pos[:]
        # self._effort_control[env_ids, :] = torch.zeros_like(pos)
        
        # self.dof_pos[env_ids, :] = pos
        # self.dof_vel[env_ids, :] = 0.0
        # print("diablo_dof_targets shape : ", self.diablo_dof_targets.shape)
        # print("dof_pos shape : ", pos.shape)

        indexes = self.indexes_sim_diablo[env_ids]
        # multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()

        # self.gym.set_dof_position_target_tensor_indexed(
        #     self.sim,
        #     gymtorch.unwrap_tensor(self.root_state),
        #     gymtorch.unwrap_tensor(indexes),
        #     len(env_ids),
        # )
        # self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
        #                                                 gymtorch.unwrap_tensor(self._effort_control),
        #                                                 gymtorch.unwrap_tensor(multi_env_ids_int32),
        #                                                 len(multi_env_ids_int32))
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(indexes), len(env_ids))        
        
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(indexes),
            len(env_ids),
        )
        self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
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
        # print("multi_env_ids_int32_len: ", len(multi_env_ids_int32))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew' + key] = self.episode_sums[key][env_ids]
            self.episode_sums[key][env_ids] = 0


    def pre_physics_step(self, actions):
        #check actions has nan
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone().to(self.device)

        if torch.isnan(actions).any():
            print("WARNING: actions contain NaN inside step()")


        # TODO: Implement gripper control from actions (currently not used)
        # Split arm and gripper command
        # u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        self.actions[..., 0:2] = 0.0  # Remove head action
        self.actions[..., [6,7,8,9,10,11,12,13,14,15,16]] = 0.0 # Remove left griper action
        self.actions[..., [18,19]] = 0
        for i in self.wheelIdx:
            self.actions[:,i] *= self.wheel_action_scale

        # Fixed Knee Joint 
        for i in self.kneeIdx:
            self.actions[:,i] = 0

        # Sync Hip Joints
        self.actions[:,self.hipIdx[0]] = 0
        self.actions[:,self.hipIdx[1]] = 0

        # # # Arm movements
        # for i in self.elIdx:
        #     self.actions[:,i] = 0
        self.actions[:,self.elIdx[1]] = 0
        # self.actions[:,self.shoRollIdx[0]] = 0
        self.actions[:,self.shoRollIdx[1]] = 0
        # self.actions[:,self.shoPitchIdx[0]] = 0
        self.actions[:,self.shoPitchIdx[1]] = 0

        targets = (torch.tensor(self.action_scale, device=self.device) * torch.tensor(self.actions, device=self.device)) + torch.tensor(self.default_dof_pos, device=self.device)
        
        # SET POS and VEL targets
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor((targets)))
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor((targets)))


        # targets = self.action_scale * self.actions + self.diablo_default_dof_pos
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))
        
    

        # u_arm, u_gripper = self.actions[:, :-12], self.actions[:, -12:-3]
        # u_fingers = torch.zeros((self.num_envs, 9), dtype=torch.float32, device=self.device)
 
        # u_fingers[:, :9] = u_gripper

        # # Access the last 9 upper limits directly as a tensor or list
        self.gripper_upper_limits = self.diablo_dof_upper_limits[-9:]

        # # 夾緊並設置目標位置
        # self.diablo_dof_targets = tensor_clamp(
        #     targets, self.diablo_dof_lower_limits, self.diablo_dof_upper_limits
        # )


        # # Deploy actions
        # self.gym.set_dof_position_target_tensor(
        #     self.sim, gymtorch.unwrap_tensor(self.diablo_dof_targets)
        # )

    def quat_to_euler(q):
        x, y, z, w = q.unbind(-1)

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * (torch.pi / 2),
            torch.asin(sinp)
        )

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return torch.stack([roll, pitch, yaw], dim=-1)



    def post_physics_step(self):
        
        self.progress_buf += 1


        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        if torch.isnan(self.obs_buf).any():
            print("WARNING: obs_buf contains NaN")

        self.compute_observations()
        self.compute_reward()


#####################################################################
###=========================jit functions=========================###
#####################################################################


# @torch.jit.script
def compute_diablo_reward(
    root_robot: torch.Tensor,
    commands: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    base_quat: torch.Tensor,
    torques: torch.Tensor,
    actions: torch.Tensor,
    contact_forces: torch.Tensor,
    knee_indices: torch.Tensor,
    progress_buf: torch.Tensor,
    ####################
    rew_scales: Dict[str, float],
    lin_vel_scale: float,
    ang_vel_scale: float,
    base_index: int,
    max_episode_length: float,

    end_effector_pos: torch.Tensor,
    target_pos: torch.Tensor,
    prev_actions: torch.Tensor,
    dof_vel: torch.Tensor,
    std_dist: float,
    weight_dist: float,
    weight_dist_tanh: float,
    weight_action_rate: float,
    weight_joint_vel: float,
    weightGripper: float,
    log_reward_tensor: torch.Tensor,
    diablo_dof_lower_limits: torch.Tensor,
    diablo_dof_upper_limits: torch.Tensor,
    gripper_upper_limits: torch.Tensor,
    u_fingers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor,torch.Tensor,torch.Tensor]:
    
    # ---- Grasp 部分 ----
    position_err = torch.norm(end_effector_pos - target_pos, dim=-1)
    position_err_tanh = 1 - torch.tanh(position_err * std_dist)

    gripped = torch.where(position_err[:, None] <= 0.1, gripper_upper_limits, diablo_dof_lower_limits[-9:][None, :])

    action_rate = torch.sum(torch.square(actions - prev_actions), dim=1)
    joint_vel = torch.sum(torch.square(dof_vel), dim=1)

    total_position_err = position_err * (-(weight_dist**2)) 
    total_position_err_tanh = position_err_tanh * weight_dist_tanh
    total_action_rate = action_rate * weight_action_rate
    total_joint_vel = joint_vel * weight_joint_vel

    success_gripper = torch.zeros(position_err.shape[0], device=position_err.device, dtype=torch.float)

    success_gripper[
        (gripped == gripper_upper_limits).all(dim=1)
    ] = 1.0

    success_gripper[
        (position_err > 0.1) | (gripped == diablo_dof_lower_limits[-9:]).all(dim=1)
    ] = -1.0

    reach_goal = (position_err <= 0.01)

    base_euler = quat_to_euler(base_quat)
    base_lin_vel = quat_rotate_inverse(base_quat, base_lin_vel)
    base_ang_vel = quat_rotate_inverse(base_quat, base_ang_vel)

    lin_vel_error = torch.sum(torch.square(commands[:, :1] - base_lin_vel[:, :1]), dim=1)
    ang_vel_error = torch.square(commands[:, 1] - base_ang_vel[:, 2])
    rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * rew_scales["lin_vel_xy"]
    rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * rew_scales["ang_vel_z"]

    # 摔倒判斷：Pitch 或 Roll 超過 ±120° 就算跌倒
    fallen = (torch.abs(base_euler[:,1]) > (2 * torch.pi / 3)) | (torch.abs(base_euler[:,0]) > (2 * torch.pi / 3))

    # print("fallen: ", fallen)
    # 總獎勵
    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]
    rew_fall = fallen.float() * rew_scales["fall"]  # fall scale 通常給個 -50 或更小

    # 懲罰 Pitch 過大（仰角）
    rew_pitch = torch.square(base_euler[:,1])
    rew_pitch[rew_pitch < 0.99] = 0
    rew_pitch = rew_pitch * rew_scales["pitch"]
    balance_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_torque + rew_pitch + rew_fall


    # # ---- 合併 Reward ----
    # grasp_reward = total_position_err + total_position_err_tanh - total_action_rate - total_joint_vel + success_gripper * weightGripper
    # balance_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_torque + rew_pitch

    log_reward_tensor[0] = total_position_err
    log_reward_tensor[1] = total_position_err_tanh
    log_reward_tensor[2] = total_action_rate
    log_reward_tensor[3] = total_joint_vel

    # 加總並 clip
    rewards = balance_reward
    rewards = torch.clip(rewards, 0., None)

    # ---- Reset 條件 ----


    reset_buf = torch.norm(contact_forces[:, base_index, :], dim=1) > 0.1
    reset_buf|= torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)
    reset_buf[fallen] = 1
    time_out = torch.where(progress_buf >= max_episode_length - 1,torch.ones_like(reset_buf),reset_buf,)
    reset_buf = reset_buf | time_out
    # reset_buf = torch.where(
    #     reach_goal, torch.ones_like(reset_buf), reset_buf
    # )

    print("knee_indices", knee_indices)
    # print("Base force norm:", contact_forces[:, base_index, :])
    # print("Knee force norm:", torch.norm(contact_forces[:, knee_indices, :], dim=2))

    # print("baseindex: ", base_index)
    # print("knee_indices: ", knee_indices)

    print("contact_forces.shape : ", contact_forces.shape)
    knee_forces = torch.norm(contact_forces[:, knee_indices, :], dim=2)
    print("Max knee force per env:", torch.max(torch.norm(contact_forces[:, knee_indices, :], dim=2), dim=1).values)

    return rewards, reset_buf, log_reward_tensor,gripper_upper_limits
# u_fingers
