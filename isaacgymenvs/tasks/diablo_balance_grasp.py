"""
DiabloBalanceGrasp
==================
Simultaneous wheeled-balance and arm-grasping task for the DIABLO robot.

Flow:
  Phase 0  BALANCE_APPROACH  –  robot stabilises its inverted-pendulum body and
                                 drives toward the table object.
  Phase 1  GRASP             –  robot keeps balance while the arm reaches,
                                 grasps and lifts the object.

The transition from Phase 0 → Phase 1 is latched once the robot has maintained
|pitch| < balanced_pitch_threshold for balance_min_steps consecutive steps AND
is within grasp_approach_dist of the object handle.

Action space (9):
  [0]   height command   → IK leg height  [h_min, h_max] m
  [1]   pitch  command   → IK leg pitch   [-p_max, +p_max] rad  (balance lean)
  [2]   left  wheel vel  → [-wheel_vel_max, +wheel_vel_max] rad/s
  [3]   right wheel vel  → [-wheel_vel_max, +wheel_vel_max] rad/s
  [4‥7] right arm joints (4, delta from default)
  [8]   gripper open / close

Observation space (63):
  progress(1), base_lin_vel_body(3), base_ang_vel(3), base_quat(4),
  base_height_norm(1), leg_pos_norm(4), leg_vel(4), wheel_vel(2),
  arm_pos_norm(4), arm_vel(4), eef_pos(3), eef_rot(4),
  object_pos(3), object_rot(4), handle_pos(3),
  eef→handle(3), robot→handle(3), phase(1), prev_actions(9)
"""

import math
import os
from typing import Tuple

import numpy as np
import torch
from isaacgym import gymapi, gymtorch

from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import (
    quat_apply, quat_from_euler_xyz, quat_mul, quat_rotate_inverse,
    axisangle2quat, to_torch, tensor_clamp,
)


# ── helper ────────────────────────────────────────────────────────────────────

def _quat_to_pitch_roll(q: torch.Tensor):
    """Return (pitch, roll) scalars per env from quaternion (x,y,z,w)."""
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll  = torch.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    sinp  = torch.clamp(2*(qw*qy - qz*qx), -1.0, 1.0)
    pitch = torch.asin(sinp)
    return pitch, roll


# ══════════════════════════════════════════════════════════════════════════════

class DiabloBalanceGrasp(VecTask):

    PHASE_APPROACH = 0
    PHASE_GRASP    = 1

    # Physical constants (match diablo_graspcustom3.py)
    R_WHEEL = 0.08
    L1 = L2 = 0.14
    L_MAX   = 0.28   # L1 + L2
    # Offset from URDF base_link origin down to the hip joint plane.
    # Inferred from diablo22 spawn height (0.45m) vs our IK model height (0.29m).
    BODY_OFFSET = 0.16

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id,
                 headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        env_cfg  = cfg["env"]

        self.max_episode_length    = env_cfg["episodeLength"]
        self.action_scale          = env_cfg.get("actionScale",  1.5)
        self.wheel_vel_max         = env_cfg.get("wheelVelMax",  5.0)

        # Leg IK ranges
        self.h_mid   = 0.29;  self.h_range = 0.06
        self.h_min   = 0.23;  self.h_max   = 0.35
        # Set h_grasp_mid back to low height (0.39m) to use as a target for placement
        self.h_grasp_mid = self.h_min + self.BODY_OFFSET   # 0.39 m
        self.p_max   = 0.20   # ± rad

        # Phase thresholds
        self.fall_pitch_thr      = env_cfg.get("fallPitchThreshold",     0.50)
        self.balanced_pitch_thr  = env_cfg.get("balancedPitchThreshold", 0.18)
        self.balance_min_steps   = env_cfg.get("balanceMinSteps",        5)
        self.grasp_approach_dist = env_cfg.get("graspApproachDist",      0.25)

        # Reward scales
        rwd = env_cfg["rewards"]
        self.balance_scale         = rwd.get("balanceScale",       3.0)
        self.alive_bonus           = rwd.get("aliveBonus",         2.0)
        self.height_scale          = rwd.get("heightScale",        2.0)
        self.approach_scale        = rwd.get("approachScale",      2.0)
        self.dist_scale            = rwd.get("distRewardScale",    2.0)
        self.rot_scale             = rwd.get("rotRewardScale",     0.8)
        self.grasp_scale           = rwd.get("graspRewardScale",   5.0)
        self.lift_scale            = rwd.get("liftRewardScale",    15.0)
        self.success_bonus         = rwd.get("successBonus",       500.0)
        self.fall_penalty          = rwd.get("fallPenalty",        100.0)
        self.action_penalty_scale  = rwd.get("actionPenaltyScale", 0.002)

        # DOF name lists
        self.right_arm_names = ["r_sho_pitch", "r_sho_roll", "r_el", "r_wrist"]
        self.right_gripper_names = [
            "r_index_base", "r_index_middle", "r_index_tip",
            "r_mid_base",   "r_mid_middle",   "r_mid_tip",
            "r_thumb_base", "r_thumb_middle",  "r_thumb_tip",
        ]
        self.dof_indices = {}

        n = env_cfg["numEnvs"]
        dev = sim_device

        # Per-env buffers
        self.phase_buf      = torch.zeros(n, dtype=torch.long,    device=dev)
        self.balance_timer  = torch.zeros(n, dtype=torch.float32, device=dev)
        self.actions        = torch.zeros((n, 9), dtype=torch.float32, device=dev)
        self.prev_actions   = torch.zeros((n, 9), dtype=torch.float32, device=dev)
        self.episode_success = torch.zeros(n, dtype=torch.bool,   device=dev)

        # Platform data
        self.platform_pos_tensor = torch.zeros((n, 3), dtype=torch.float32, device=dev)

        # Statistics
        self.total_attempts  = 0
        self.total_successes = 0

        self.up_axis     = "z"
        self.up_axis_idx = 2
        self.states      = {}
        self.debug_mode  = False

        # Override counts (also declared in YAML, but we pin them here)
        self.cfg["env"]["numActions"]      = 9
        self.cfg["env"]["numObservations"] = 69

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
            self.gym.viewer_camera_look_at(
                self.viewer, None,
                gymapi.Vec3(0.5, -2.5, 1.5),
                gymapi.Vec3(0.0,  0.0, 0.35),
            )

        self._acquire_tensors()
        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[..., 1]

        self.gripper_upper_limits = self.dof_upper_limits[self.dof_indices["right_gripper"]]
        self.gripper_lower_limits = self.dof_lower_limits[self.dof_indices["right_gripper"]]

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._refresh_tensors()

    # ── sim creation ──────────────────────────────────────────────────────────

    def create_sim(self):
        self.sim_params.up_axis   = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0.0
        self.sim_params.gravity.y = 0.0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id,
            self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(
            self.num_envs,
            self.cfg["env"]["envSpacing"],
            int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        pp = gymapi.PlaneParams()
        pp.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, pp)

    # ── env / asset creation ──────────────────────────────────────────────────

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3( spacing,  spacing, spacing)

        asset_root  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        diablo_file = self.cfg["env"]["asset"]["assetFileNamediablo"]

        # ── Robot (free-base: can fall, must balance) ──────────────────────
        ao = gymapi.AssetOptions()
        ao.fix_base_link           = False   # KEY: physics body
        ao.collapse_fixed_joints   = False
        ao.disable_gravity         = False
        ao.armature                = 0.01
        ao.default_dof_drive_mode  = gymapi.DOF_MODE_POS
        ao.use_mesh_materials      = False
        ao.flip_visual_attachments = False
        print(f"Loading {diablo_file} …")
        diablo_asset = self.gym.load_asset(self.sim, asset_root, diablo_file, ao)

        self.num_diablo_bodies = self.gym.get_asset_rigid_body_count(diablo_asset)
        self.num_diablo_shapes = self.gym.get_asset_rigid_shape_count(diablo_asset)
        self.num_dofs          = self.gym.get_asset_dof_count(diablo_asset)
        print(f"Total DOFs: {self.num_dofs}")

        dof_dict = self.gym.get_asset_dof_dict(diablo_asset)
        self.dof_indices["right_arm"]     = [dof_dict[n] for n in self.right_arm_names]
        self.dof_indices["right_gripper"] = [dof_dict[n] for n in self.right_gripper_names]
        self.dof_indices["legs"]          = [
            dof_dict["left_fake_hip_joint"],  dof_dict["left_fake_knee_joint"],
            dof_dict["right_fake_hip_joint"], dof_dict["right_fake_knee_joint"],
        ]
        self.dof_indices["wheel_l"] = dof_dict["left_wheel_joint"]
        self.dof_indices["wheel_r"] = dof_dict["right_wheel_joint"]
        print(f"Leg DOF IDs : {self.dof_indices['legs']}")
        print(f"Wheel DOF IDs: L={self.dof_indices['wheel_l']}  R={self.dof_indices['wheel_r']}")

        # DOF properties ── wheels → velocity mode; rest → position mode
        dof_props = self.gym.get_asset_dof_properties(diablo_asset)
        stiffness_map = {
            "head_pan":  1e6, "head_tilt": 1e6,
            "l_sho_pitch": 1e6, "l_sho_roll": 1e6, "l_el": 1e6, "l_wrist": 1e6,
            "r_sho_pitch": 15.0, "r_sho_roll": 20.0, "r_el": 13.82, "r_wrist": 4.55,
            "left_fake_hip_joint":  400.0, "left_fake_knee_joint":  400.0,
            "right_fake_hip_joint": 400.0, "right_fake_knee_joint": 400.0,
            "left_wheel_joint":  0.0,   # velocity mode — stiffness unused
            "right_wheel_joint": 0.0,
        }
        damping_map = {
            "head_pan": 100., "head_tilt": 100.,
            "l_sho_pitch": 100., "l_sho_roll": 100., "l_el": 100., "l_wrist": 100.,
            "r_sho_pitch": 0.3, "r_sho_roll": 0.4, "r_el": 0.1, "r_wrist": 0.002,
            "left_fake_hip_joint":  50., "left_fake_knee_joint":  50.,
            "right_fake_hip_joint": 50., "right_fake_knee_joint": 50.,
            "left_wheel_joint":  20.,   # velocity-tracking P gain
            "right_wheel_joint": 20.,
        }

        self.dof_lower_limits = []
        self.dof_upper_limits = []
        for i in range(self.num_dofs):
            name = self.gym.get_asset_dof_name(diablo_asset, i)
            if name in ("left_wheel_joint", "right_wheel_joint"):
                dof_props["driveMode"][i] = gymapi.DOF_MODE_VEL
            else:
                dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][i] = stiffness_map.get(name, 700.0)
            dof_props["damping"][i]   = damping_map.get(name, 50.0)
            if name in self.right_gripper_names:
                dof_props["stiffness"][i] = 20.0
                dof_props["damping"][i]   = 0.5
                dof_props["effort"][i]    = 0.2
            self.dof_lower_limits.append(float(dof_props["lower"][i]))
            self.dof_upper_limits.append(float(dof_props["upper"][i]))

        self.dof_lower_limits = to_torch(self.dof_lower_limits, device=self.device)
        self.dof_upper_limits = to_torch(self.dof_upper_limits, device=self.device)

        # ── Table (identical to diablo_graspcustom3) ───────────────────────
        tao = gymapi.AssetOptions(); tao.fix_base_link = True
        table_asset  = self.gym.create_box(self.sim, 0.3, 0.5, 0.01, tao)
        n_tab_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        n_tab_shapes = self.gym.get_asset_rigid_shape_count(table_asset)

        TABLE_POS      = gymapi.Vec3(0.25, 0.0, 0.55)
        TABLE_HALF_HEIGHT = 0.005      # create_box height=0.01 → half=0.005
        self.table_surface_z = TABLE_POS.z + TABLE_HALF_HEIGHT  # 0.555

        # ── Object asset ───────────────────────────────────────────────────
        obj_file = self.cfg["env"]["object"]["objectRoot"]
        oao = gymapi.AssetOptions()
        oao.fix_base_link      = False
        oao.use_mesh_materials = True
        oao.mesh_normal_mode   = gymapi.COMPUTE_PER_VERTEX
        oao.override_com       = True
        oao.override_inertia   = True
        oao.vhacd_enabled      = True
        oao.vhacd_params       = gymapi.VhacdParams()
        oao.vhacd_params.resolution = 1000
        self.object_asset = self.gym.load_asset(self.sim, asset_root, obj_file, oao)
        n_obj_bodies  = self.gym.get_asset_rigid_body_count(self.object_asset)
        n_obj_shapes  = self.gym.get_asset_rigid_shape_count(self.object_asset)
        self.object_height      = self.cfg["env"]["object"]["objectHeight"]
        self.object_half_height = self.cfg["env"]["object"]["objectHalfHeight"]

        self.obj_start_z = self.table_surface_z + self.object_half_height + 0.005
        self.initial_object_z = torch.full(
            (num_envs,), self.obj_start_z, dtype=torch.float32, device=self.device)

        # ── Platform asset ────────────────────────────────────────────────
        pao = gymapi.AssetOptions(); pao.fix_base_link = True
        platform_asset = self.gym.create_box(self.sim, 0.1, 0.1, 0.01, pao)
        n_plat_bodies = self.gym.get_asset_rigid_body_count(platform_asset)
        n_plat_shapes = self.gym.get_asset_rigid_shape_count(platform_asset)

        total_bodies = self.num_diablo_bodies + n_tab_bodies + n_obj_bodies + n_plat_bodies
        total_shapes = self.num_diablo_shapes + n_tab_shapes + n_obj_shapes + n_plat_shapes

        # ── Static poses ───────────────────────────────────────────────────
        table_pose   = gymapi.Transform()
        table_pose.p = TABLE_POS
        table_pose.r = gymapi.Quat(0, 0, 0, 1)

        robot_init_pose   = gymapi.Transform()
        robot_init_pose.p = gymapi.Vec3(-0.55, 0.0, 0.29)
        robot_init_pose.r = gymapi.Quat(0, 0, 0, 1)

        obj_init_pose   = gymapi.Transform()
        obj_init_pose.p = gymapi.Vec3(TABLE_POS.x, 0.0, self.obj_start_z)
        obj_init_pose.r = gymapi.Quat(0, 0, 1, 0)

        platform_init_pose   = gymapi.Transform()
        platform_init_pose.p = gymapi.Vec3(0.0, 0.0, -1.0) # Hidden initially
        platform_init_pose.r = gymapi.Quat(0, 0, 0, 1)

        self.envs           = []
        self.actor_handles  = []
        self.table_handles  = []
        self.obj_handles    = []
        self.plat_handles   = []

        for i in range(num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env, total_bodies, total_shapes, True)

            # Robot
            diablo = self.gym.create_actor(env, diablo_asset, robot_init_pose, "diablo", i, -1, 0)
            self.gym.set_actor_dof_properties(env, diablo, dof_props)
            self.actor_handles.append(diablo)

            # Table
            table = self.gym.create_actor(env, table_asset, table_pose, "table", i, 1, 0)
            self.table_handles.append(table)

            # Object (random Y on table)
            obj_init_pose.p.y = float(np.random.uniform(-0.15, 0.15))
            obj_actor = self.gym.create_actor(env, self.object_asset, obj_init_pose, "object", i, 2, 0)
            self.obj_handles.append(obj_actor)

            # Platform
            plat_actor = self.gym.create_actor(env, platform_asset, platform_init_pose, "platform", i, 3, 0)
            self.gym.set_rigid_body_color(env, plat_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.4, 0.1))
            self.plat_handles.append(plat_actor)

            self.gym.end_aggregate(env)
            self.envs.append(env)

        # Global actor IDs (each env: 0=diablo, 1=table, 2=object, 3=platform)
        self.diablo_actor_ids = torch.arange(0, num_envs * 4, 4, dtype=torch.int32, device=self.device)
        self.table_actor_ids  = torch.arange(1, num_envs * 4, 4, dtype=torch.int32, device=self.device)
        self.obj_actor_ids    = torch.arange(2, num_envs * 4, 4, dtype=torch.int32, device=self.device)
        self.plat_actor_ids   = torch.arange(3, num_envs * 4, 4, dtype=torch.int32, device=self.device)

        self.eef_handle = self.gym.find_actor_rigid_body_index(
            self.envs[0], self.actor_handles[0], "panda_grip_site", gymapi.DOMAIN_ENV)
        self.handle_target_handle = self.gym.find_actor_rigid_body_index(
            self.envs[0], self.obj_handles[0], "handle_target", gymapi.DOMAIN_ENV)
        print(f"EEF body idx: {self.eef_handle}  |  Handle-target body idx: {self.handle_target_handle}")

        self.init_data()

    # ── data initialisation ───────────────────────────────────────────────────

    def init_data(self):
        self._acquire_tensors()

        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float32, device=self.device)
        arm_def = np.radians([0, -60, -80, 0])
        for idx, val in zip(self.dof_indices["right_arm"], arm_def):
            self.default_dof_pos[idx] = float(val)
        self.default_dof_pos[self.dof_indices["right_gripper"]] = \
            self.dof_lower_limits[self.dof_indices["right_gripper"]]

        # Phase-1 arm default: body is at 0.39m (h_grasp_mid), but graspcustom3 arm pose
        # was tuned for 0.25m body height. Reduce sho_roll from -60° to -10° so the natural
        # EEF height (≈body + 0.02m) matches the table object height (≈0.41m).
        self.arm_default_grasp = self.default_dof_pos.clone()
        arm_grasp_def = np.radians([0, -10, -55, 0])
        for idx, val in zip(self.dof_indices["right_arm"], arm_grasp_def):
            self.arm_default_grasp[idx] = float(val)

    def _acquire_tensors(self):
        self.root_state_tensor       = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor        = self.gym.acquire_dof_state_tensor(self.sim)
        self.rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.root_state       = gymtorch.wrap_tensor(self.root_state_tensor)
        self.dof_state        = gymtorch.wrap_tensor(self.dof_state_tensor)
        self.rigid_body_state = gymtorch.wrap_tensor(self.rigid_body_state_tensor)
        self.rigid_body_pos   = self.rigid_body_state[:, :3].view(self.num_envs, -1, 3)
        self.rigid_body_rot   = self.rigid_body_state[:, 3:7].view(self.num_envs, -1, 4)

    def _refresh_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._update_states()

    def _update_states(self):
        base_quat    = self.root_state[self.diablo_actor_ids, 3:7]
        world_lin_vel = self.root_state[self.diablo_actor_ids, 7:10]
        world_ang_vel = self.root_state[self.diablo_actor_ids, 10:13]
        body_lin_vel  = quat_rotate_inverse(base_quat, world_lin_vel)

        self.states.update({
            "base_pos":          self.root_state[self.diablo_actor_ids, :3],
            "base_quat":         base_quat,
            "base_lin_vel":      body_lin_vel,
            "base_ang_vel":      world_ang_vel,
            "eef_pos":           self.rigid_body_pos[:, self.eef_handle],
            "eef_rot":           self.rigid_body_rot[:, self.eef_handle],
            "object_pos":        self.root_state[self.obj_actor_ids, :3],
            "object_rot":        self.root_state[self.obj_actor_ids, 3:7],
            "handle_target_pos": self.rigid_body_pos[:, self.handle_target_handle],
            "handle_target_rot": self.rigid_body_rot[:, self.handle_target_handle],
            "dof_pos":           self.dof_pos,
            "dof_vel":           self.dof_vel,
        })

    # ── observations ──────────────────────────────────────────────────────────

    def compute_observations(self):
        self._refresh_tensors()

        base_pos    = self.states["base_pos"]
        base_quat   = self.states["base_quat"]
        base_lv     = self.states["base_lin_vel"]
        base_av     = self.states["base_ang_vel"]
        eef_pos     = self.states["eef_pos"]
        eef_rot     = self.states["eef_rot"]
        obj_pos     = self.states["object_pos"]
        obj_rot     = self.states["object_rot"]
        handle_pos  = self.states["handle_target_pos"]
        dof_pos     = self.states["dof_pos"]
        dof_vel     = self.states["dof_vel"]

        delta = self.dof_upper_limits - self.dof_lower_limits + 1e-6
        dof_pos_norm = (dof_pos - self.dof_lower_limits) / delta

        leg_ids = self.dof_indices["legs"]
        arm_ids = self.dof_indices["right_arm"]
        wl_id   = self.dof_indices["wheel_l"]
        wr_id   = self.dof_indices["wheel_r"]

        leg_pos_norm  = dof_pos_norm[:, leg_ids]
        leg_vel       = dof_vel[:, leg_ids]
        wheel_vel     = torch.stack([dof_vel[:, wl_id], dof_vel[:, wr_id]], dim=-1)
        arm_pos_norm  = dof_pos_norm[:, arm_ids]
        arm_vel       = dof_vel[:, arm_ids]

        height_norm   = ((base_pos[:, 2] - (self.h_mid + self.BODY_OFFSET)) / self.h_range).unsqueeze(-1)
        eef_to_handle   = handle_pos - eef_pos
        robot_to_handle = handle_pos - base_pos
        phase_obs = self.phase_buf.float().unsqueeze(-1)

        # Object bottom for placement
        obj_up_local = torch.tensor([0., 0., 1.], device=self.device).repeat(self.num_envs, 1)
        world_obj_up = quat_apply(obj_rot, obj_up_local)
        obj_bottom_pos = obj_pos - self.object_half_height * world_obj_up
        rel_bottom_to_plat = self.platform_pos_tensor - obj_bottom_pos

        o = self.obs_buf
        idx = 0
        def w(t, n):
            nonlocal idx
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            o[:, idx:idx + n] = t
            idx += n

        w(self.progress_buf.float() / self.max_episode_length, 1)  # 1
        w(base_lv,         3)   # 4
        w(base_av,         3)   # 7
        w(base_quat,       4)   # 11
        w(height_norm,     1)   # 12
        w(leg_pos_norm,    4)   # 16
        w(leg_vel,         4)   # 20
        w(wheel_vel,       2)   # 22
        w(arm_pos_norm,    4)   # 26
        w(arm_vel,         4)   # 30
        w(eef_pos,         3)   # 33
        w(eef_rot,         4)   # 37
        w(obj_pos,         3)   # 40
        w(obj_rot,         4)   # 44
        w(handle_pos,      3)   # 47
        w(eef_to_handle,   3)   # 50
        w(robot_to_handle, 3)   # 53
        w(phase_obs,       1)   # 54
        w(self.prev_actions, 9) # 63
        w(self.platform_pos_tensor, 3) # 66
        w(rel_bottom_to_plat, 3)       # 69

        assert idx == 69, f"obs mismatch {idx}"
        self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0, posinf=0.0, neginf=0.0)
        return self.obs_buf

    # ── reward ────────────────────────────────────────────────────────────────

    def compute_reward(self):
        (
            self.rew_buf[:], self.reset_buf[:],
            bal_rew, alive_rew, height_rew, approach_rew,
            dist_rew, rot_rew, grasp_rew, lift_rew, trans_rew, place_rew, release_rew, retreat_rew, orient_rew,
            grasp_bal_bonus, success_rew, total_pen,
            smooth_pen, energy_pen, arm_motion_pen, fall_pen, braking_pen,
            is_success, is_fallen, is_balanced, is_grasping, is_lifted,
        ) = compute_balance_grasp_reward(
            reset_buf         = self.reset_buf,
            progress_buf      = self.progress_buf,
            actions           = self.actions,
            prev_actions      = self.prev_actions,
            base_pos          = self.states["base_pos"],
            base_quat         = self.states["base_quat"],
            base_ang_vel      = self.states["base_ang_vel"],
            eef_pos           = self.states["eef_pos"],
            eef_rot           = self.states["eef_rot"],
            handle_pos        = self.states["handle_target_pos"],
            handle_rot        = self.states["handle_target_rot"],
            object_pos        = self.states["object_pos"],
            object_rot        = self.states["object_rot"],
            initial_object_z  = self.initial_object_z,
            platform_pos      = self.platform_pos_tensor,
            phase_buf         = self.phase_buf,
            num_envs          = self.num_envs,
            max_episode_length= self.max_episode_length,
            h_mid             = self.h_mid + self.BODY_OFFSET,
            h_grasp_mid       = self.h_grasp_mid,
            fall_pitch_thr    = self.fall_pitch_thr,
            object_half_height= self.object_half_height,
            balance_scale     = self.balance_scale,
            alive_bonus       = self.alive_bonus,
            height_scale      = self.height_scale,
            approach_scale    = self.approach_scale,
            dist_scale        = self.dist_scale,
            rot_scale         = self.rot_scale,
            grasp_scale       = self.grasp_scale,
            lift_scale        = self.lift_scale,
            success_bonus     = self.success_bonus,
            fall_penalty      = self.fall_penalty,
            action_penalty_scale = self.action_penalty_scale,
        )

        # ── Positive reward components ─────────────────────────────────────
        self.extras["rewards/balance"]        = bal_rew.mean()
        self.extras["rewards/alive"]          = alive_rew.mean()
        self.extras["rewards/height"]         = height_rew.mean()
        self.extras["rewards/approach"]       = approach_rew.mean()
        self.extras["rewards/dist"]           = dist_rew.mean()
        self.extras["rewards/rot"]            = rot_rew.mean()
        self.extras["rewards/grasp"]          = grasp_rew.mean()
        self.extras["rewards/lift"]           = lift_rew.mean()
        self.extras["rewards/transport"]      = trans_rew.mean()
        self.extras["rewards/placement"]      = place_rew.mean()
        self.extras["rewards/release"]        = release_rew.mean()
        self.extras["rewards/retreat"]        = retreat_rew.mean()
        self.extras["rewards/orient"]         = orient_rew.mean()
        self.extras["rewards/grasp_balance"]  = grasp_bal_bonus.mean()
        self.extras["rewards/success"]        = success_rew.mean()
        # ── Penalty components (positive = larger penalty) ─────────────────
        self.extras["rewards/pen_total"]      = total_pen.mean()
        self.extras["rewards/pen_smooth"]     = smooth_pen.mean()
        self.extras["rewards/pen_energy"]     = energy_pen.mean()
        self.extras["rewards/pen_arm"]        = arm_motion_pen.mean()
        self.extras["rewards/pen_fall"]       = fall_pen.mean()
        self.extras["rewards/pen_braking"]    = braking_pen.mean()
        # ── Behavioural metrics ────────────────────────────────────────────
        self.extras["metrics/fall_rate"]      = is_fallen.float().mean()
        self.extras["metrics/balanced_rate"]  = is_balanced.float().mean()
        self.extras["metrics/grasping_rate"]  = is_grasping.float().mean()
        self.extras["metrics/lifted_rate"]    = is_lifted.float().mean()
        self.extras["metrics/phase"]          = self.phase_buf.float().mean()
        self.extras["metrics/balance_timer"]  = self.balance_timer.mean()

        self.episode_success |= is_success

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset_idx(self, env_ids):
        num = len(env_ids)
        R, l1, l2 = self.R_WHEEL, self.L1, self.L2

        # Random initial height and small pitch perturbation
        h_tgt = 0.29 + (torch.rand(num, device=self.device) - 0.5) * 0.04
        p_tgt = (torch.rand(num, device=self.device) - 0.5) * 0.08   # ±0.04 rad
        y_tgt = (torch.rand(num, device=self.device) - 0.5) * 0.60   # ±0.30 rad yaw

        # IK → leg angles
        L0        = torch.clamp((h_tgt - R) / torch.cos(p_tgt), 0.01, l1 + l2)
        cos_alpha = torch.clamp((l1**2 + l2**2 - L0**2) / (2*l1*l2), -1.0, 1.0)
        tgt_knee  = math.pi - torch.acos(cos_alpha)
        cos_beta  = torch.clamp(L0 / (2*l1), -1.0, 1.0)
        tgt_hip   = -p_tgt - torch.acos(cos_beta)

        leg_ids = self.dof_indices["legs"]
        self.dof_pos[env_ids, leg_ids[0]] = tgt_hip
        self.dof_pos[env_ids, leg_ids[1]] = tgt_knee
        self.dof_pos[env_ids, leg_ids[2]] = tgt_hip
        self.dof_pos[env_ids, leg_ids[3]] = tgt_knee

        arm_grip = self.dof_indices["right_arm"] + self.dof_indices["right_gripper"]
        self.dof_pos[env_ids[:, None], arm_grip] = self.default_dof_pos[arm_grip]
        self.dof_vel[env_ids, :] = 0.0

        # Robot starts 0.5–0.7 m in front of table (negative X, facing table)
        start_x = -(0.50 + torch.rand(num, device=self.device) * 0.20)
        start_y =  (torch.rand(num, device=self.device) - 0.5) * 0.20
        bx = L0 * torch.sin(p_tgt) * torch.cos(y_tgt) + start_x
        by = L0 * torch.sin(p_tgt) * torch.sin(y_tgt) + start_y

        self.root_state[self.diablo_actor_ids[env_ids], 0] = bx
        self.root_state[self.diablo_actor_ids[env_ids], 1] = by
        self.root_state[self.diablo_actor_ids[env_ids], 2] = h_tgt + self.BODY_OFFSET
        robot_quat = quat_from_euler_xyz(torch.zeros_like(p_tgt), p_tgt, y_tgt)
        self.root_state[self.diablo_actor_ids[env_ids], 3:7]  = robot_quat
        self.root_state[self.diablo_actor_ids[env_ids], 7:13] = 0.0

        # Object: random position on table surface
        obj_x = torch.clamp(0.25 + (torch.rand(num, device=self.device) - 0.5) * 0.14, 0.18, 0.32)
        obj_y = torch.clamp((torch.rand(num, device=self.device) - 0.5) * 0.20, -0.10, 0.10)
        self.root_state[self.obj_actor_ids[env_ids], 0] = obj_x
        self.root_state[self.obj_actor_ids[env_ids], 1] = obj_y
        self.root_state[self.obj_actor_ids[env_ids], 2] = self.obj_start_z
        aa = torch.zeros(num, 3, device=self.device)
        aa[:, 2] = math.pi + (torch.rand(num, device=self.device) - 0.5) * 0.5
        init_rot = torch.tensor([0., 0., 0., 1.], device=self.device).unsqueeze(0).repeat(num, 1)
        self.root_state[self.obj_actor_ids[env_ids], 3:7]  = quat_mul(axisangle2quat(aa), init_rot)
        self.root_state[self.obj_actor_ids[env_ids], 7:13] = 0.0

        # Apply
        d_ids = self.diablo_actor_ids[env_ids].to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(d_ids), len(d_ids))

        # Platform: random comfortable zone relative to robot
        # local: X ≈ 0.25, Y ≈ -0.22
        lp_x = 0.25 + (torch.rand(num, device=self.device) - 0.5) * 0.05
        lp_y = -0.22 + (torch.rand(num, device=self.device) - 0.5) * 0.05
        lp_z = torch.zeros(num, device=self.device)
        lp_pos = torch.stack([lp_x, lp_y, lp_z], dim=-1)
        wp_off = quat_apply(robot_quat, lp_pos)
        self.platform_pos_tensor[env_ids] = torch.stack([bx, by, torch.zeros_like(bx)], dim=-1) + wp_off
        self.platform_pos_tensor[env_ids, 2] = -1.0 # Hidden

        all_ids = torch.cat([
            self.diablo_actor_ids[env_ids],
            self.obj_actor_ids[env_ids],
            self.plat_actor_ids[env_ids]
        ]).to(torch.int32)
        self.root_state[self.plat_actor_ids[env_ids], :3] = self.platform_pos_tensor[env_ids]
        self.root_state[self.plat_actor_ids[env_ids], 3:7] = torch.tensor([0., 0., 0., 1.], device=self.device).repeat(num, 1)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(all_ids), len(all_ids))

        # Reset counters
        self.progress_buf[env_ids]    = 0
        self.reset_buf[env_ids]       = 0
        self.phase_buf[env_ids]       = self.PHASE_APPROACH
        self.balance_timer[env_ids]   = 0.0
        self.actions[env_ids]         = 0.0
        self.prev_actions[env_ids]    = 0.0
        self.episode_success[env_ids] = False

    # ── physics step ──────────────────────────────────────────────────────────

    def pre_physics_step(self, actions):
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone().to(self.device)

        N  = self.num_envs
        pos_targets = self.default_dof_pos.unsqueeze(0).repeat(N, 1).clone()
        vel_targets = torch.zeros(N, self.num_dofs, device=self.device)

        # ── Leg IK: actions [0]=h, [1]=p ──────────────────────────────────
        h_cmd = self.actions[:, 0]
        p_cmd = self.actions[:, 1]
        h_tgt = torch.clamp(self.h_mid + h_cmd * self.h_range, self.h_min, self.h_max)
        p_tgt = torch.clamp(p_cmd * self.p_max, -self.p_max, self.p_max)
        L0    = torch.clamp((h_tgt - self.R_WHEEL) / torch.cos(p_tgt), 0.01, self.L_MAX)
        c_a   = torch.clamp((self.L1**2 + self.L2**2 - L0**2) / (2*self.L1*self.L2), -1.0, 1.0)
        knee  = math.pi - torch.acos(c_a)
        c_b   = torch.clamp(L0 / (2*self.L1), -1.0, 1.0)
        hip   = -p_tgt - torch.acos(c_b)

        leg_ids = self.dof_indices["legs"]
        pos_targets[:, leg_ids[0]] = hip
        pos_targets[:, leg_ids[1]] = knee
        pos_targets[:, leg_ids[2]] = hip
        pos_targets[:, leg_ids[3]] = knee

        # ── Wheels: actions [2]=left, [3]=right (velocity mode) ───────────
        vel_targets[:, self.dof_indices["wheel_l"]] = self.actions[:, 2] * self.wheel_vel_max
        vel_targets[:, self.dof_indices["wheel_r"]] = self.actions[:, 3] * self.wheel_vel_max

        # ── Arm: actions [4‥7] with phase-dependent default ──────────────
        arm_ids = self.dof_indices["right_arm"]
        # Phase 0: default [0°,−60°,−80°,0°] (arm tucked, body 0.45m)
        # Phase 1: default [0°,−10°,−80°,0°] (arm near-horizontal, body 0.39m → EEF ≈ 0.41m = table)
        phase_grasp_mask = (self.phase_buf == self.PHASE_GRASP).unsqueeze(1).expand(-1, 4)
        arm_base = torch.where(
            phase_grasp_mask,
            self.arm_default_grasp[arm_ids].unsqueeze(0).expand(N, -1),
            self.default_dof_pos[arm_ids].unsqueeze(0).expand(N, -1),
        )
        arm_tgt = self.action_scale * self.actions[:, 4:8] + arm_base
        pos_targets[:, arm_ids] = arm_tgt

        # In Phase 0 bias arm back to Phase-0 default (balance focus) - Reduced bias to 0.4
        phase0_mask = (self.phase_buf == self.PHASE_APPROACH).unsqueeze(1).expand(-1, 4)
        pos_targets[:, arm_ids] = torch.where(
            phase0_mask,
            0.6 * arm_tgt + 0.4 * self.default_dof_pos[arm_ids],
            arm_tgt,
        )

        # ── Gripper: action [8] with proximity heuristic ──────────────────
        eef_pos    = self.states["eef_pos"]
        handle_pos = self.states["handle_target_pos"]
        dist       = torch.linalg.norm(eef_pos - handle_pos, dim=-1)
        should_close = (dist <= 0.025) & (self.actions[:, 8] >= 0.0) & \
                       (self.phase_buf == self.PHASE_GRASP)
        g_ids = self.dof_indices["right_gripper"]
        fing_tgt = torch.where(
            should_close.unsqueeze(1).expand(-1, 9),
            self.gripper_upper_limits.expand(N, 9),
            self.gripper_lower_limits.expand(N, 9),
        )
        pos_targets[:, g_ids] = fing_tgt

        pos_targets = tensor_clamp(pos_targets, self.dof_lower_limits, self.dof_upper_limits)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_targets))
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(vel_targets))

    def post_physics_step(self):
        self.progress_buf += 1
        self._refresh_tensors()

        # Reveal platform when object is lifted
        obj_heights = self.states["object_pos"][:, 2] - self.initial_object_z
        should_appear = (obj_heights > 0.02)
        # Lower platform surface to 0.40m (below table 0.555m)
        # target_z is center, surface is target_z + 0.005. So center = 0.395.
        target_z = 0.395
        
        # Only update if hidden and should appear
        reveal_mask = should_appear & (self.platform_pos_tensor[:, 2] < 0)
        if reveal_mask.any():
            self.platform_pos_tensor[reveal_mask, 2] = target_z
            self.root_state[self.plat_actor_ids[reveal_mask], 2] = target_z
            
            p_ids = self.plat_actor_ids[reveal_mask].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.root_state),
                gymtorch.unwrap_tensor(p_ids), len(p_ids))

        # ── Phase transition logic ─────────────────────────────────────────
        pitch, roll = _quat_to_pitch_roll(self.states["base_quat"])
        is_bal = (torch.abs(pitch) < self.balanced_pitch_thr) & \
                 (torch.abs(roll)  < self.balanced_pitch_thr * 1.5)
        self.balance_timer = torch.where(is_bal, self.balance_timer + 1,
                                          torch.zeros_like(self.balance_timer))

        base_pos   = self.states["base_pos"]
        base_quat  = self.states["base_quat"]
        handle_pos = self.states["handle_target_pos"]
        # Pre-grasp position: where the base should be for the right arm to reach the handle.
        # Golden zone (from graspcustom3): 0.22 m forward, 0.15 m to the right in robot frame.
        fwd_w = quat_apply(base_quat,
                           torch.tensor([1., 0., 0.], device=self.device).unsqueeze(0).repeat(self.num_envs, 1))
        rt_w  = quat_apply(base_quat,
                           torch.tensor([0.,-1., 0.], device=self.device).unsqueeze(0).repeat(self.num_envs, 1))
        pregrasp_xy = handle_pos[:, :2] - 0.22 * fwd_w[:, :2] - 0.15 * rt_w[:, :2]
        dist_to_pregrasp = torch.norm(pregrasp_xy - base_pos[:, :2], dim=-1)
        in_range   = dist_to_pregrasp < self.grasp_approach_dist
        ready      = in_range & (self.balance_timer >= self.balance_min_steps)

        # Latch: approach → grasp, never revert until reset
        self.phase_buf = torch.where(
            (self.phase_buf == self.PHASE_APPROACH) & ready,
            torch.ones_like(self.phase_buf) * self.PHASE_GRASP,
            self.phase_buf,
        )

        # ── Episode resets ────────────────────────────────────────────────
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.total_attempts  += len(env_ids)
            self.total_successes += self.episode_success[env_ids].sum().item()
            sr = self.total_successes / max(self.total_attempts, 1)
            self.extras["metrics/success_rate"] = sr
            self.reset_idx(env_ids)
            self.episode_success[env_ids] = False

        self.compute_observations()
        self.compute_reward()


# ══════════════════════════════════════════════════════════════════════════════
# Master reward function
# ══════════════════════════════════════════════════════════════════════════════

def compute_balance_grasp_reward(
    reset_buf, progress_buf, actions, prev_actions,
    base_pos, base_quat, base_ang_vel,
    eef_pos, eef_rot, handle_pos, handle_rot,
    object_pos, object_rot, initial_object_z,
    platform_pos,
    phase_buf,
    num_envs: int, max_episode_length: float,
    h_mid: float, h_grasp_mid: float, fall_pitch_thr: float, object_half_height: float,
    balance_scale: float, alive_bonus: float, height_scale: float,
    approach_scale: float, dist_scale: float, rot_scale: float,
    grasp_scale: float, lift_scale: float, success_bonus: float,
    fall_penalty: float, action_penalty_scale: float,
):
    dev = base_pos.device

    # ── Extract pitch and roll ─────────────────────────────────────────────
    qx, qy, qz, qw = base_quat[:,0], base_quat[:,1], base_quat[:,2], base_quat[:,3]
    roll  = torch.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    sinp  = torch.clamp(2*(qw*qy - qz*qx), -1.0, 1.0)
    pitch = torch.asin(sinp)
    pitch_abs = torch.abs(pitch)
    roll_abs  = torch.abs(roll)

    # ── 1. Balance reward (always active) ─────────────────────────────────
    # Gaussian bell: +6 when perfectly upright, smoothly decays
    bal_rew  = torch.exp(-15.0 * pitch**2) + torch.exp(-15.0 * roll**2)
    ang_speed = torch.norm(base_ang_vel, dim=-1)
    bal_rew  = bal_rew - 0.3 * torch.clamp(ang_speed, 0.0, 5.0)
    bal_rew  = bal_rew * balance_scale

    # ── 2. Alive bonus (per step without falling) ──────────────────────────
    is_balanced = (pitch_abs < fall_pitch_thr) & (roll_abs < fall_pitch_thr)
    alive_rew   = torch.where(is_balanced,
                              torch.full_like(bal_rew, alive_bonus),
                              torch.zeros_like(bal_rew))

    # ── 3. Height regulation ───────────────────────────────────────────────
    # Dynamically change height target based on sub-phase
    obj_up_local = torch.tensor([0., 0., 1.], device=dev).repeat(num_envs, 1)
    w_obj_up = quat_apply(object_rot, obj_up_local)
    obj_bottom_pos = object_pos - object_half_height * w_obj_up
    dist_xy_to_plat = torch.norm(obj_bottom_pos[:, :2] - platform_pos[:, :2], p=2, dim=-1)

    # Logic:
    # Phase 0: Stay at 0.45m
    # Phase 1 (Grasping Table): Stay at 0.45m
    # Phase 1 (Placing on Plat): Squat to 0.39m (h_grasp_mid)
    is_placing = (phase_buf == 1) & (dist_xy_to_plat < 0.20)
    
    height_target = torch.where(
        is_placing,
        torch.full_like(base_pos[:, 2], h_grasp_mid),
        torch.full_like(base_pos[:, 2], h_mid),
    )
    height_err = torch.abs(base_pos[:, 2] - height_target)
    height_rew = torch.exp(-20.0 * height_err**2) * height_scale

    # ── 4. Approach reward (Phase 0 only) ─────────────────────────────────
    # Guide robot to pre-grasp base position: where the right arm can reach the handle.
    # Golden zone (from graspcustom3): 0.22 m forward + 0.15 m right in robot frame.
    fwd_w = quat_apply(base_quat,
                       torch.tensor([1., 0., 0.], device=dev).unsqueeze(0).expand(num_envs, -1))
    rt_w  = quat_apply(base_quat,
                       torch.tensor([0.,-1., 0.], device=dev).unsqueeze(0).expand(num_envs, -1))
    pregrasp_xy      = handle_pos[:, :2] - 0.22 * fwd_w[:, :2] - 0.15 * rt_w[:, :2]
    dist_to_pregrasp = torch.norm(pregrasp_xy - base_pos[:, :2], dim=-1)
    # Reciprocal reward for longer range reach (1.0 at dist=0, 0.4 at dist=1m)
    approach_rew     = (1.0 / (1.0 + 1.5 * dist_to_pregrasp)) * approach_scale
    approach_rew     = torch.where(phase_buf == 0, approach_rew,
                                   torch.zeros_like(approach_rew))

    # In Phase 0 lightly penalise arm motion so robot focuses on balance
    arm_actions    = actions[:, 4:8]
    arm_motion_pen = torch.sum(arm_actions**2, dim=-1) * 0.1
    arm_motion_pen = torch.where(phase_buf == 0, arm_motion_pen,
                                  torch.zeros_like(arm_motion_pen))

    # ── 5. EEF reaching (Phase 1 only) ────────────────────────────────────
    d_eef    = torch.norm(eef_pos - handle_pos, p=2, dim=-1)
    # Sharper peak for precision: 1.0 at 0m, 0.5 at 0.22m (1/1.04), 0.1 at 0.67m
    dist_rew = 1.0 / (1.0 + 20.0 * d_eef**2)
    dist_rew = dist_rew * dist_scale
    # Give 10% even in Phase 0 so arm begins conditioning early
    dist_rew = torch.where(phase_buf == 1, dist_rew, dist_rew * 0.1)

    # ── 6. Gripper orientation alignment (Phase 1 only) ───────────────────
    g_fwd = torch.tensor([0., 0., -1.], device=dev).repeat(num_envs, 1)
    g_up  = torch.tensor([1., 0.,  0.], device=dev).repeat(num_envs, 1)
    h_in  = torch.tensor([1., 0.,  0.], device=dev).repeat(num_envs, 1)
    h_up  = torch.tensor([0., 0.,  1.], device=dev).repeat(num_envs, 1)

    ax1 = quat_apply(eef_rot, g_fwd);  ax2 = quat_apply(handle_rot, h_in)
    ax3 = quat_apply(eef_rot, g_up);   ax4 = quat_apply(handle_rot, h_up)
    dot1 = (ax1 * ax2).sum(-1)
    dot2 = (ax3 * ax4).sum(-1)
    rot_rew = 0.5 * (torch.sign(dot1)*dot1**2 + torch.sign(dot2)*dot2**2) * rot_scale
    rot_rew = torch.where(phase_buf == 1, rot_rew, torch.zeros_like(rot_rew))

    # ── 7. Grasp attempt reward ────────────────────────────────────────────
    is_aligned    = (dot1 < -0.70) & (dot2 > 0.70)
    is_close_eef  = (d_eef < 0.035) & is_aligned
    gripper_close = actions[:, 8] >= 0.0
    grasp_rew     = torch.where(
        is_close_eef & gripper_close & (phase_buf == 1),
        torch.full_like(dist_rew, grasp_scale * 2.0),
        torch.zeros_like(dist_rew))

    # ── 8. Lift reward ─────────────────────────────────────────────────────
    obj_height  = object_pos[:, 2] - initial_object_z
    is_grasping = is_close_eef & gripper_close & (phase_buf == 1)
    target_lift = 0.06  # m
    lift_rew    = torch.where(is_grasping,
                              lift_scale * torch.clamp(obj_height, 0.0, target_lift),
                              torch.zeros_like(dist_rew))

    # Keep object upright while lifting
    obj_up  = torch.tensor([0., 0., 1.], device=dev).repeat(num_envs, 1)
    w_obj_up = quat_apply(object_rot, obj_up)
    world_up = torch.tensor([0., 0., 1.], device=dev).repeat(num_envs, 1)
    obj_upright_dot = (w_obj_up * world_up).sum(-1)
    is_upright = obj_upright_dot > 0.85

    orient_rew = torch.pow(torch.clamp(obj_upright_dot, 0.0), 8) * 20.0
    orient_rew = torch.where(is_grasping & (obj_height > 0.01), orient_rew,
                              torch.zeros_like(orient_rew))

    # ── 9. Transport & Placement ─────────────────────────────────────────
    obj_bottom_pos = object_pos - object_half_height * w_obj_up
    dist_xy_to_plat = torch.norm(obj_bottom_pos[:, :2] - platform_pos[:, :2], p=2, dim=-1)
    plat_surface_z = platform_pos[:, 2] + 0.005
    dist_z_to_plat = torch.abs(obj_bottom_pos[:, 2] - plat_surface_z)
    
    is_lifted = (obj_height > 0.03)
    is_over_plat = (dist_xy_to_plat < 0.04)
    is_on_plat   = is_over_plat & (dist_z_to_plat < 0.02)
    
    trans_rew = torch.where(is_grasping & is_lifted, 30.0 * torch.exp(-5.0 * dist_xy_to_plat), torch.zeros_like(dist_rew))
    trans_rew = torch.where(~is_upright, trans_rew * 0.1, trans_rew)
    
    place_rew = 50.0 / (1.0 + dist_z_to_plat * 25.0 + dist_xy_to_plat * 40.0)
    place_rew = torch.where(is_on_plat & is_upright, torch.full_like(dist_rew, 50.0),
                            torch.where(is_grasping & is_upright & (dist_xy_to_plat < 0.1), place_rew, torch.zeros_like(dist_rew)))

    # ── 10. Grasp-balance synergy bonus ────────────────────────────────────
    # Extra reward for keeping balance WHILE actively grasping/lifting
    grasp_bal_bonus = torch.where(
        is_grasping & is_balanced & is_lifted,
        torch.full_like(dist_rew, 5.0),
        torch.zeros_like(dist_rew))

    # ── 11. Release & Final Success ───────────────────────────────────────
    # Relaxed thresholds to kickstart learning
    gripper_open = actions[:, 8] < 0.0 # Any negative value triggers opening
    eef_dist_to_obj = torch.norm(eef_pos - object_pos, p=2, dim=-1)
    
    # Require being on platform and upright before rewarding release
    is_releasing = is_on_plat & is_upright & gripper_open
    release_rew = torch.where(is_releasing, torch.full_like(dist_rew, 100.0), torch.zeros_like(dist_rew))
    
    # Retreat reward: encourage hand to move away after dropping
    retreat_rew = torch.where(is_releasing, torch.clamp(eef_dist_to_obj, 0.0, 0.1) * 800.0, torch.zeros_like(dist_rew))

    # Urgency penalty: Stop them from just hovering and holding
    # Small penalty for every step they are over the platform but NOT open
    hover_pen = torch.where(is_over_plat & is_lifted & (~gripper_open), 
                            torch.full_like(dist_rew, 2.0), torch.zeros_like(dist_rew))

    # Success condition: placed + upright + balanced + gripper open + small hand away
    is_success  = is_on_plat & is_upright & is_balanced & gripper_open & (eef_dist_to_obj > 0.035)
    success_rew = torch.where(is_success,
                               torch.full_like(dist_rew, success_bonus),
                               torch.zeros_like(dist_rew))

    # ── 12. Base Braking (Phase 1 only) ───────────────────────────────────
    # Penalise base movement once in Grasp phase to prevent crashing into table
    wheel_cmds = actions[:, 2:4]
    braking_pen = torch.where(phase_buf == 1, torch.sum(wheel_cmds**2, dim=-1) * 2.0, torch.zeros_like(dist_rew))

    # ── Penalties ─────────────────────────────────────────────────────────
    action_delta   = actions - prev_actions
    smooth_pen     = torch.sum(action_delta**2, dim=-1) * action_penalty_scale
    energy_pen     = torch.sum(actions**2,      dim=-1) * action_penalty_scale * 0.3
    time_pen       = torch.full_like(bal_rew, 1.5)

    # Grace period: don't trigger fall reset during physics settling (first 20 steps)
    is_fallen  = ((pitch_abs > fall_pitch_thr) | (roll_abs > fall_pitch_thr)) & (progress_buf >= 20)
    fall_pen   = torch.where(is_fallen,
                              torch.full_like(bal_rew, fall_penalty),
                              torch.zeros_like(bal_rew))

    total_pen = smooth_pen + energy_pen + time_pen + arm_motion_pen + fall_pen + braking_pen + hover_pen

    # ── Total ──────────────────────────────────────────────────────────────
    rewards = (
        bal_rew + alive_rew + height_rew + approach_rew +
        dist_rew + rot_rew + grasp_rew + lift_rew + orient_rew +
        trans_rew + place_rew + release_rew + retreat_rew +
        grasp_bal_bonus + success_rew -
        total_pen
    )

    # ── Reset conditions ───────────────────────────────────────────────────
    reset_buf = torch.where(is_fallen,  torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(is_success, torch.ones_like(reset_buf), reset_buf)

    # Don't reset if dropped ON platform
    obj_dropped = (object_pos[:, 2] < 0.20) & (~is_grasping) & (~is_on_plat)
    reset_buf   = torch.where(obj_dropped, torch.ones_like(reset_buf), reset_buf)

    reset_buf = torch.where(
        progress_buf >= max_episode_length - 1,
        torch.ones_like(reset_buf), reset_buf)

    return (
        rewards, reset_buf,
        bal_rew, alive_rew, height_rew, approach_rew,
        dist_rew, rot_rew, grasp_rew, lift_rew, trans_rew, place_rew, release_rew, retreat_rew, orient_rew,
        grasp_bal_bonus, success_rew, total_pen,
        smooth_pen, energy_pen, arm_motion_pen, fall_pen, braking_pen,
        is_success, is_fallen, is_balanced, is_grasping, is_lifted,
    )
