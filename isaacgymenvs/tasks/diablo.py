# Copyright (c) 2018-2022, NVIDIA Corporation
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

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
from typing import Tuple, Dict
from torch.utils.tensorboard import SummaryWriter





class diablo(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales['heading_scale'] = self.cfg["env"]["learn"]["headingScale"]
        self.rew_scales['up_scale'] = self.cfg["env"]["learn"]["upScale"]
        self.rew_scales['air_time'] = self.cfg["env"]["learn"]["feetAirTimeRewardScale"]
        self.rew_scales['syns_hip'] = self.cfg["env"]["learn"]["syncronizeHipRewardScale"]
        self.rew_scales['no_fly'] = self.cfg["env"]["learn"]["noflyRewardScale"]
        self.rew_scales['stand_scale'] = self.cfg["env"]["learn"]["standRewardScale"]
        self.rew_scales['action_rate'] = self.cfg["env"]["learn"]["actionRateRewardScale"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        # self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        # self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        # self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]
        # self.command_heading_range = self.cfg["env"]["randomCommandVelocityRanges"]["heading"]

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

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        self.cfg["env"]["numObservations"] = 93
        self.cfg["env"]["numActions"] = 28

        print(self.cfg)
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
       

        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)


        actors_per_env = 4
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, actors_per_env, 13)
        # self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)
        # print("pos",self.dof_pos)
        # print("vel",self.dof_vel)

        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.gym.refresh_dof_force_tensor(self.sim)

        

        # default_joint_angles = {
           
        #     # Other joint angles
        #     'head_pan': 0,
        #     'head_tilt': 0,
        #     'l_el': 1.4,
        #     'l_index_base': 0,
        #     'l_index_middle': 0,
        #     'l_index_tip': 0,
        #     'l_mid_base': 0,
        #     'l_mid_middle': 0,
        #     'l_mid_tip': 0,
        #     'l_sho_pitch': 0,
        #     'l_sho_roll': 1,
        #     'l_thumb_base': 0,
        #     'l_thumb_middle': 0,
        #     'l_thumb_tip': 0,
        #     'l_wrist': 0,
        #     'r_el': -1.4,
        #     'r_index_base': 0,
        #     'r_index_middle': 0,
        #     'r_index_tip': 0,
        #     'r_mid_base': 0,
        #     'r_mid_middle': 0,
        #     'r_mid_tip': 0,
        #     'r_sho_pitch': 0,
        #     'r_sho_roll': -1,
        #     'r_thumb_base': 0,
        #     'r_thumb_middle': 0,
        #     'r_thumb_tip': 0,
        #     'r_wrist': 0
        # }





        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        # self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        # self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        # self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]

        # 根据你的关节角度字典创建一个包含默认关节角度的张量
        # default_joint_angles_tensor = torch.tensor(list(default_joint_angles.values()), dtype=torch.float, device=self.device)

        # 使用这个张量的第一个元素来填充相同形状的张量
        # self.default_dof_pos = torch.full_like(self.dof_pos, default_joint_angles_tensor[0])
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)


        # # 根据你的关节角度字典创建一个包含默认关节角度的张量
        # default_joint_angles_tensor = torch.tensor(list(default_joint_angles.values()), dtype=torch.float, device=self.device)

        # # # 使用这个张量来填充相同形状的张量
        # self.default_dof_pos = torch.full_likefull

        # self.default_dof_pos = torch.full_like(self.dof_pos, default_joint_angles, dtype=torch.float, device=self.device, requires_grad=False)
        self.op3_init_xy = torch.tensor(self.base_init_state[0:2], device=self.device)
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # initialize some data used later on
        self.extras = {}
        # self.initial_root_states = self.root_states.clone()
        self.initial_root_states = self.root_states.clone()
        # self.initial_root_states[:, 7:13] = 0  # set lin_vel and ang_vel to 0
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.reward_step = torch.zeros((self.num_envs), device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_left_contacts = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device, requires_grad=False)
        self.step_episode = 0
    
        # reward episode sums
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"progress" : torch_zeros(), 
                            "upright"    :  torch_zeros(), 
                            "torques"  :    torch_zeros(), 
                            "stand_still":  torch_zeros(), 
                            "no_fly"       :torch_zeros(),
                            "air_time"  :   torch_zeros(), 
                            "contact_force":torch_zeros(), 
                            "action_rate"  :torch_zeros(),
                            "alive"       :torch_zeros(),
                            "syns_hip"  :   torch_zeros(), 
                            "orient":torch_zeros(),} 

        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()        
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.targets = to_torch([0, 10, 0], device=self.device).repeat((self.num_envs, 1))
        self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.potentials = to_torch([-10./self.dt], device=self.device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()
        self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/323Assembase_limit3/urdf/323Assembase_limit3.urdf"


        
        
        # Create table asset
        table_pos = [0.0, 0.0, 0.13]
        table_thickness = 0.25
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[.5, .5, table_thickness], table_opts)
        
        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [0.10, 0.0, 0.28 ]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.3, 0.5, table_stand_height], table_opts)


        self.cubeA_size = 0.050
        # Create cube  asset
        cubeA_opts = gymapi.AssetOptions()
        cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
        cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

        # convex decomposition with custom params
        asset_options1 = gymapi.AssetOptions()
        asset_options1.vhacd_enabled = True
        # asset_options1.fix_base_link = Truecd 
        asset_options1.vhacd_params = gymapi.VhacdParams()
        asset_options1.vhacd_params.resolution = 300000
        asset1 = self.gym.load_asset(self.sim, asset_root, "urdf/ycb/011_banana/011_banana.urdf", asset_options1)
        
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.density = 0.001
       

        asset_options.angular_damping = 0.4
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        # asset_options1.stiffness = 3
        asset_options.disable_gravity = False
        
        op3_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(op3_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(op3_asset)

        op3_assets =[]
        for _ in range(self.num_agents):
            op3_asset =self.gym.load_asset(self.sim,asset_root,asset_file,asset_options)
            op3_assets.append(op3_asset)
        dof_props =self.gym.get_asset_dof_properties(op3_assets[0])

        start_pose = gymapi.Transform()
       
    
        start_pose.r = gymapi.Quat.from_euler_zyx(*self.base_init_state[3:6])
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)



        # Define start pose for table
        table_start_pose = gymapi.Transform()
        
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])

        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # cubeA_start_pose = gymapi.Transform()
        # cubeA_start_pose.p = gymapi.Vec3(.05, 0.0, 0.3)
        # cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        mug_start_pose = gymapi.Transform()
        mug_start_pose.p = gymapi.Vec3(.05, 0.0, 0.3)
        mug_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        body_names = self.gym.get_asset_rigid_body_names(op3_asset)
        self.dof_names = self.gym.get_asset_dof_names(op3_asset)
        print(self.dof_names)
        extremity_name = "SHANK" if asset_options.collapse_fixed_joints else "ank"
        feet_names = [s for s in body_names if extremity_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if "knee" in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(op3_asset)
        for i in range(self.num_dof): #---> 0-5
            dof_props['driveMode'][i] = self.cfg["env"]["urdfAsset"]["defaultDofDriveMode"] #gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd
            dof_props['effort'][i] = 100.0 #self.Kd

        # dof_props['driveMode'][8] = self.cfg["env"]["urdfAsset"]["defaultDofDriveMode"] #gymapi.DOF_MODE_POS
        # dof_props['stiffness'][8] = self.cfg["env"]["control"]["stiffness"] #self.Kp
        # dof_props['damping'][8] = self.cfg["env"]["control"]["damping"] #self.Kd

        # class control( diablo.control ):
        # # PD Drive parameters:jdsh=
        #     stiffness = { 'head_pan': 200.0, 'head_tilt': 200.0,
        #                 'l_sho_pitch': 200., 'r_sho_pitch': 200., 'l_sho_roll': 150.,'r_sho_roll':150,
        #                 'l_el': 100.,'r_el':100,'l_wrist':50,'r_wrist':50,'l_index_base':30,'r_index_base':30,'l_index_middle':20,'r_index_middle':20,'l_index_tip':10,'r_index_tip':10,'l_mid_base':30,'r_mid_base':30,'l_mid_middle':20,'r_mid_middle':20,'l_mid_tip':10,"r_mid_tip":10,'l_thumb_base':30,'r_thumb_base':30,'l_thumb_middle':20,'r_thumb_middle':20,'l_thumb_tip':10,'r_thumb_tip':10}  # [N*m/rad]
        #     damping = { 'head_pan': 6.0, 'head_tilt': 6.0,
        #                 'l_sho_pitch': 6., 'r_sho_pitch': 6., 'l_sho_roll': 4.5,'r_sho_roll':4.5,
        #                 'l_el': 3.,'r_el':3,'l_wrist':1.5,'r_wrist':1.5,'l_index_base':0.3,'r_index_base':0.3,'l_index_middle':0.2,'r_index_middle':0.2,'l_index_tip':0.1,'r_index_tip':0.1,'l_mid_base':0.3,'r_mid_base':0.3,'l_mid_middle':0.2,'r_mid_middle':0.2,'l_mid_tip':0.1,"r_mid_tip":0.1,'l_thumb_base':0.3,'r_thumb_base':0.3,'l_thumb_middle':0.2,'r_thumb_middle':0.2,'l_thumb_tip':0.1,'r_thumb_tip':0.1}  # [N*m*s/rad]     # [N*m*s/rad]
        # # action scale: target angle = actionScale * action + defaultAngle
        #     action_scale = 0.5
        # # decimation: Number of control action updates @ sim DT per policy DT
        #     decimation = 4



        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.op3_handles = []
        self.table_handles = []
        self.table_stand_handles = []
        self.mug_handles = []
        self.envs = []

        



        robot_idx = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 0, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",i, 0, 0)

            op3_handle = self.gym.create_actor(env_ptr, op3_asset, start_pose, "op3", i, 2, 0)
            robot_idx.append(self.gym.get_actor_index(env_ptr, op3_handle, gymapi.DOMAIN_SIM))
            self.gym.set_actor_dof_properties(env_ptr, op3_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, op3_handle)
            # self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0)
            mug = self.gym.create_actor(env_ptr, asset1, mug_start_pose, "mug", i, 3, 0)
            # self.gym.set_rigid_body_color(env_ptr, mug, 0, gymapi.MESH_VISUAL, cubeA_color)


            self.envs.append(env_ptr)
            self.op3_handles.append(op3_handle)
            self.table_handles.append(table_actor)
            self.table_stand_handles.append(table_stand_actor)
            self.mug_handles.append(mug)

        self.robot_indices = to_torch(robot_idx, dtype=torch.int32, device=self.device)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.op3_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.op3_handles[0], knee_names[i])
        
     
        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.op3_handles[0], "base-frame-link")



        # self.gym.set_actor_dof_properties(env_ptr, op3_handle, dof_props)
        # dof_states = self.gym.get_actor_dof_states(env_ptr, op3_handle, gymapi.STATE_ALL)
        #         # Modify dof_states as needed, similar to your single agent example
        #         # For example, if you have predefined angles for each DOF
        # for dof_name, dof_angle in self.named_default_joint_angles.items():
        #         dof_index = self.gym.find_asset_dof_index(op3_assets[0], dof_name)
        #         dof_states['pos'][dof_index] = dof_angle
        #         self.gym.set_actor_dof_states(env_ptr, op3_handle, dof_states, gymapi.STATE_ALL)

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        self.actions[..., 0:13] = 0.0  # Remove head action
        # self.actions[..., 3:13] = 0.0
       
        # targets = self.default_dof_pos
        targets = self.action_scale * self.actions + self.default_dof_pos
        # print(targets)

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))
        
        self.last_actions[:] = self.actions[:]

    def post_physics_step(self):
        self.progress_buf += 1
        self.step_episode += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.prev_torques = self.torques.clone()
        self.compute_observations()
        self.compute_reward(self.actions)

    def _reward_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.1
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt

        return rew_airTime

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  300).clip(min=0.), dim=1)
    
    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
    
    def compute_reward(self, actions):

        reward_airtime = self._reward_air_time() * self.rew_scales["air_time"]
        reward_step    = self._reward_feet_contact_forces()
        reward_no_fly  = self._reward_no_fly() * self.rew_scales["no_fly"]
        
    

        self.rew_buf[:], self.reset_buf[:] = compute_op3_reward(
            # tensors
            self.root_states,
            self.commands,
            self.prev_torques,
            self.torques,
            self.contact_forces,
            self.knee_indices,
            self.progress_buf,
            # Dict
            self.rew_scales,
            # other
            self.base_index,
            self.max_episode_length,
            self.targets,
            self.heading_vec,
            self.up_vec,
            self.inv_start_rot,
            reward_airtime,
            reward_step,
            reward_no_fly,
            self.reset_buf,
            self.actions,
            self.last_actions,
            self.dof_pos,
            self.default_dof_pos,
            self.op3_init_xy,
            self.episode_sums,
            self.gravity_vec,
            self.dt,
            self.potentials,
            self.prev_potentials,
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        
        self.obs_buf[:],self.potentials[:], self.prev_potentials[:] = compute_op3_observations(self.root_states,
                                                           self.targets,
                                                           self.dof_pos,
                                                           self.default_dof_pos,
                                                           self.dof_vel,
                                                           self.gravity_vec,
                                                           self.actions,
                                                           # scales
                                                           self.lin_vel_scale,
                                                           self.ang_vel_scale,
                                                           self.dof_pos_scale,
                                                           self.dof_vel_scale,
                                                           self.dt,
                                                           self.potentials,
        )

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(-0.5, 0.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(self.robot_indices), len(self.robot_indices))

        # self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        # self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        # self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

        to_target = self.targets[env_ids] - self.initial_root_states[env_ids, 3, 0:3]
        to_target[:, self.up_axis_idx] = 0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            # print(key)
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.

        self.last_actions[env_ids] = 0.
        self.feet_air_time[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    
#####################################################################
###=========================jit functions=========================###
#####################################################################
# @torch.jit.script
def compute_op3_reward(
    # tensors
    root_states,
    commands,
    prev_torques,
    torques,
    contact_forces,
    knee_indices,
    episode_lengths,
    # Dict
    rew_scales: Dict[str, float],
    # other
    base_index: int,
    max_episode_length: int,
    targets,
    vec0,
    vec1,
    inv_start_rot,
    rew_airtime,
    rew_step,
    rew_no_fly,
    reset_buf,
    actions,
    last_actions,
    dof_pos,
    default_dof_pos,
    op3_init_state,
    episode_sums: Dict[str, torch.Tensor],
    gravity_vec,
    dt:float,
    potentials,
    prev_potentials,
    ):
    # Prepare quantities (TODO: return from obs?)
    quat = root_states[:,0, 3:7]
    ang_vel = quat_rotate_inverse(quat, root_states[:,0, 10:13])

    position = root_states[:,0, 0:3]
    rotation = root_states[:,0, 3:7]

    z_position = root_states[:,0, 2]
    op3_z_position = torch.clamp((z_position - 0.0) / (0.27 - 0.0), 0.0, 1.0)

    #Get vector of heading and upright
    to_target = targets - position
    to_target[:, 2] = 0
    target_dirs = normalize(to_target)

    num_envs = position.shape[0]
    torso_quat = quat_mul(rotation, inv_start_rot)

    heading_vec = get_basis_vector(torso_quat, vec0).view(num_envs, 3)
    heading_proj = torch.bmm(heading_vec.view(num_envs, 1, 3), target_dirs.view(num_envs, 3, 1)).view(num_envs)

    # Reward from direction headed
    heading_weight_tensor = torch.ones_like(heading_proj) * rew_scales["heading_scale"]
    heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, rew_scales["heading_scale"] * heading_proj / 0.8)
    up_vec = get_basis_vector(torso_quat, vec1).view(num_envs, 3)
    up_proj = up_vec[:, 2]

    # # distance from hand to the cubeA
    # d = torch.norm(states["cubeA_pos_relative"], dim=-1)
    # d_lf = torch.norm(states["cubeA_pos"] - states["eef_lf_pos"], dim=-1)
    # d_rf = torch.norm(states["cubeA_pos"] - states["eef_rf_pos"], dim=-1)
    # dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)

    # Aligning up axis of ant and environment
    rew_up = torch.zeros_like(heading_reward)
    rew_up = torch.where(up_proj > 0.93, rew_up + rew_scales['up_scale'], rew_up)

    # reward for duration of staying alive
    to_target = targets - position
    to_target[:, 2] = 0.0
    alive_reward = torch.ones_like(potentials) * 0.5
    progress_reward = (potentials - prev_potentials) * 5

    # projected gravity reward
    projected_gravity = quat_rotate_inverse(quat, gravity_vec)
    rew_orient = torch.sum(torch.square(projected_gravity[:, :2]), dim=1) * -0
    
    # Pinalty for Torque
    rew_torque = torch.sum(torch.abs(prev_torques - torques), dim=1) * rew_scales["torque"]
    
    # Pinalty for leg of the robot
    leg_position_pinalty = torch.sum(torch.abs(dof_pos[:, [2,3,4,5,6,7,12,13,14,15,16,17]]-\
                                               default_dof_pos[:,[2,3,4,5,6,7,12,13,14,15,16,17]]),dim=1)
    rew_syns = leg_position_pinalty * rew_scales["syns_hip"]
    
    # Action rate (reward)
    rew_action_rate = torch.sum(torch.square(last_actions - actions), dim=1) * rew_scales["action_rate"]

    # Pinalty still stand 
    rew_stand_still = torch.sum(torch.abs(dof_pos - default_dof_pos), dim=1) * (torch.norm(commands[:, :2], dim=1) < 0.1) * rew_scales["stand_scale"]

    ang_vel_error = torch.square(targets[:, 2] - ang_vel[:, 2])
    rew_ang_vel_z = torch.exp(-ang_vel_error/0.1) * 0.5
    #Total Reward 
    total_reward = progress_reward + alive_reward + rew_torque + rew_up + rew_airtime + rew_step +\
                  rew_no_fly + rew_stand_still + rew_action_rate + rew_syns + rew_orient + rew_ang_vel_z
    total_reward = torch.clip(total_reward, 0., None)

    # Out of Bound
    reset_buf[episode_lengths >= max_episode_length] = 1
    reset_buf = torch.where(up_proj < 0.88, torch.ones_like(reset_buf), reset_buf)
    reset_buf[episode_lengths == 0] = 0

    
    return total_reward.detach(), reset_buf

# @torch.jit.script
def compute_op3_observations(root_states,
                                target,
                                dof_pos,
                                default_dof_pos,
                                dof_vel,
                                gravity_vec,
                                actions,
                                lin_vel_scale,
                                ang_vel_scale,
                                dof_pos_scale,
                                dof_vel_scale,
                                dt,
                                potentials,
                                ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, Tensor) -> Tuple[Tensor, Tensor, Tensor]

    torso_position = root_states[:,0,0:3]
    to_target = target - torso_position
    to_target[:, 2] = 0
    
    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    base_quat = root_states[:,0,3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:,0,7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:,0,10:13]) * ang_vel_scale
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

    obs = torch.cat((base_lin_vel,
                     base_ang_vel,
                     projected_gravity,
                     dof_pos_scaled,
                     dof_vel*dof_vel_scale,
                     actions
                     ), dim=-1)
    
    return obs,potentials,prev_potentials_new