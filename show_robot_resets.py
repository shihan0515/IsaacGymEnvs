#!/usr/bin/env python3
"""
Diablo reset-configuration display for thesis figures.

12 environments in a 4x3 grid (4 per row, 3 rows):
  Row 0: Height variation  h in {0.23, 0.27, 0.31, 0.35} m  (pitch=0, yaw=0)
  Row 1: Pitch variation   p in {-0.20, -0.07, +0.07, +0.20} rad  (h=0.29 m, yaw=0)
  Row 2: Yaw variation     y in {-0.20, -0.07, +0.07, +0.20} rad  (h=0.29 m, pitch=0)

Usage (from IsaacGymEnvs/ directory):
    python show_robot_resets.py
    python show_robot_resets.py --pipeline cpu

Camera controls (IsaacGym viewer):
    Left-drag  : orbit   |  Right-drag : pan
    Scroll     : zoom    |  Close window to exit
"""

import math
import os
import sys

import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
import torch


# ── IK helpers ─────────────────────────────────────────────────────────────────

R_WHEEL = 0.08   # wheel radius (m)
L1      = 0.14   # upper leg length (m)
L2      = 0.14   # lower leg length (m)


def compute_leg_ik(h: float, p: float):
    """
    Compute leg joint angles matching the reset_idx IK in diablo_graspcustom3.py.

    Args:
        h: target body height above ground (m)
        p: pitch angle of robot base (rad)

    Returns:
        (hip_rad, knee_rad, L0_m)
    """
    L0 = (h - R_WHEEL) / math.cos(p)
    L0 = max(0.01, min(L0, L1 + L2))

    cos_alpha = (L1**2 + L2**2 - L0**2) / (2 * L1 * L2)
    cos_alpha = max(-1.0, min(1.0, cos_alpha))
    knee = math.pi - math.acos(cos_alpha)

    cos_beta = L0 / (2 * L1)
    cos_beta = max(-1.0, min(1.0, cos_beta))
    hip = -p - math.acos(cos_beta)

    return hip, knee, L0


def make_base_transform(h: float, p: float, y: float) -> gymapi.Transform:
    """
    Return the actor Transform for the robot base at body height h,
    pitch p, and yaw y, with the corrective XY offset so the wheels
    land on the ground plane (Z=0).
    """
    _, _, L0 = compute_leg_ik(h, p)

    # XY offset: projection of virtual leg on the ground plane
    bx = L0 * math.sin(p) * math.cos(y)
    by = L0 * math.sin(p) * math.sin(y)

    # Quaternion from quat_from_euler_xyz(roll=0, pitch=p, yaw=y)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    # qx = -sy*sp, qy = cy*sp, qz = sy*cp, qw = cy*cp
    quat = gymapi.Quat(-sy * sp, cy * sp, sy * cp, cy * cp)

    t = gymapi.Transform()
    t.p = gymapi.Vec3(bx, by, h)
    t.r = quat
    return t


# ── Preset configurations ──────────────────────────────────────────────────────

H_MID = 0.29   # reference height used for pitch / yaw rows

CONFIGS = [
    # ── Row 0: Height variation (pitch=0, yaw=0) ──────────────────────────────
    {"h": 0.23, "p":  0.00, "y":  0.00},
    {"h": 0.27, "p":  0.00, "y":  0.00},
    {"h": 0.31, "p":  0.00, "y":  0.00},
    {"h": 0.35, "p":  0.00, "y":  0.00},
    # ── Row 1: Pitch variation (h=0.29 m, yaw=0) ──────────────────────────────
    {"h": H_MID, "p": -0.20, "y":  0.00},
    {"h": H_MID, "p": -0.07, "y":  0.00},
    {"h": H_MID, "p": +0.07, "y":  0.00},
    {"h": H_MID, "p": +0.20, "y":  0.00},
    # ── Row 2: Yaw variation (h=0.29 m, pitch=0) ──────────────────────────────
    {"h": H_MID, "p":  0.00, "y": -0.20},
    {"h": H_MID, "p":  0.00, "y": -0.07},
    {"h": H_MID, "p":  0.00, "y": +0.07},
    {"h": H_MID, "p":  0.00, "y": +0.20},
]

NUM_PER_ROW = 4    # envs per row in the grid

# Right arm default pose (same as init_data in diablo_graspcustom3.py)
ARM_NAMES   = ["r_sho_pitch", "r_sho_roll", "r_el", "r_wrist"]
ARM_DEFAULT = np.radians([0, -60, -80, 0])

LEG_NAMES = [
    "left_fake_hip_joint",  "left_fake_knee_joint",
    "right_fake_hip_joint", "right_fake_knee_joint",
]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    gym  = gymapi.acquire_gym()
    args = gymutil.parse_arguments(description="Diablo reset-config display")

    # ── Sim params ─────────────────────────────────────────────────────────────
    sp = gymapi.SimParams()
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sp.dt       = 1.0 / 60.0
    sp.substeps = 2
    sp.use_gpu_pipeline = False

    if args.physics_engine == gymapi.SIM_PHYSX:
        sp.physx.num_position_iterations = 8
        sp.physx.num_velocity_iterations = 1
        sp.physx.contact_offset = 0.005
        sp.physx.rest_offset    = 0.0

    sim = gym.create_sim(
        args.compute_device_id, args.graphics_device_id,
        args.physics_engine, sp
    )
    if sim is None:
        print("ERROR: create_sim failed"); sys.exit(1)

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("ERROR: create_viewer failed"); sys.exit(1)

    # Overview camera: elevated diagonal view covering the full 4x3 grid.
    # With env_size=2.0 m and 4 cols x 3 rows, the grid spans roughly
    # X: 0..8 m, Y: 0..6 m.  Adjust as needed via viewer mouse controls.
    gym.viewer_camera_look_at(
        viewer, None,
        gymapi.Vec3(-1.0, -4.0, 5.0),   # camera position
        gymapi.Vec3( 3.0,  2.0, 0.4),   # look-at target
    )

    # ── Diablo asset ───────────────────────────────────────────────────────────
    asset_root  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    diablo_file = (
        "urdf/diab/Diablo_ger/Part_gripper_col_rev/URDF/diablo_erc_visualize.urdf"
    )

    ao = gymapi.AssetOptions()
    ao.fix_base_link          = True    # base fixed at actor pose
    ao.collapse_fixed_joints  = False
    ao.disable_gravity        = True    # static display – no physics drift
    ao.default_dof_drive_mode = gymapi.DOF_MODE_POS
    ao.use_mesh_materials     = False
    ao.flip_visual_attachments = False
    ao.armature               = 0.01

    print(f"Loading {diablo_file} ...")
    asset    = gym.load_asset(sim, asset_root, diablo_file, ao)
    num_dofs = gym.get_asset_dof_count(asset)
    dof_dict = gym.get_asset_dof_dict(asset)

    leg_ids = [dof_dict[n] for n in LEG_NAMES]
    arm_ids = [dof_dict[n] for n in ARM_NAMES]
    print(f"Leg DOF IDs : {dict(zip(LEG_NAMES, leg_ids))}")
    print(f"Arm DOF IDs : {dict(zip(ARM_NAMES, arm_ids))}")
    print(f"Total DOFs  : {num_dofs}")

    # High-stiffness position drive to lock every joint at the target pose
    dof_props = gym.get_asset_dof_properties(asset)
    for i in range(num_dofs):
        dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
        dof_props["stiffness"][i] = 10000.0
        dof_props["damping"][i]   = 1000.0

    # ── Create environments ────────────────────────────────────────────────────
    ENV_SZ    = 2.0                           # size of each env (m)
    env_lower = gymapi.Vec3(-ENV_SZ / 2, -ENV_SZ / 2, 0.0)
    env_upper = gymapi.Vec3( ENV_SZ / 2,  ENV_SZ / 2, 2.5)

    envs, actors = [], []
    for i, cfg in enumerate(CONFIGS):
        env  = gym.create_env(sim, env_lower, env_upper, NUM_PER_ROW)
        pose = make_base_transform(cfg["h"], cfg["p"], cfg["y"])
        actor = gym.create_actor(env, asset, pose, f"diablo_{i}", i, -1, 0)
        gym.set_actor_dof_properties(env, actor, dof_props)
        envs.append(env)
        actors.append(actor)

    # ── Apply DOF state and position targets ───────────────────────────────────
    gym.prepare_sim(sim)

    # Zero-init all DOF positions and velocities
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_state = gymtorch.wrap_tensor(dof_state_tensor)   # shape: (N*num_dofs, 2)
    dof_state[:, :] = 0.0

    target_all = torch.zeros(len(CONFIGS) * num_dofs, dtype=torch.float32)

    for i, cfg in enumerate(CONFIGS):
        hip, knee, _ = compute_leg_ik(cfg["h"], cfg["p"])
        base = i * num_dofs

        # Both legs share the same hip / knee angle (mirrored kinematics)
        for j, lid in enumerate(leg_ids):
            val = float(hip) if j % 2 == 0 else float(knee)
            dof_state[base + lid, 0] = val
            target_all[base + lid]   = val

        # Right arm default pose
        for j, aid in enumerate(arm_ids):
            val = float(ARM_DEFAULT[j])
            dof_state[base + aid, 0] = val
            target_all[base + aid]   = val

    gym.set_dof_state_tensor(
        sim, gymtorch.unwrap_tensor(dof_state)
    )
    gym.set_dof_position_target_tensor(
        sim, gymtorch.unwrap_tensor(target_all)
    )

    # ── Print configuration summary ────────────────────────────────────────────
    row_info = [
        "Row 0 — Height variation  (pitch=0,  yaw=0)",
        "Row 1 — Pitch variation   (h=0.29 m, yaw=0)",
        "Row 2 — Yaw variation     (h=0.29 m, pitch=0)",
    ]
    print("\n=== Diablo Reset-Config Display ===")
    for row in range(3):
        print(f"\n{row_info[row]}:")
        for col in range(4):
            idx  = row * 4 + col
            cfg  = CONFIGS[idx]
            hip_d, knee_d, L0 = compute_leg_ik(cfg["h"], cfg["p"])
            print(
                f"  Env {idx:2d}  h={cfg['h']:.2f} m  "
                f"p={math.degrees(cfg['p']):+6.1f}°  "
                f"y={math.degrees(cfg['y']):+6.1f}°  "
                f"→  hip={math.degrees(hip_d):+6.1f}°  "
                f"knee={math.degrees(knee_d):+6.1f}°  "
                f"L0={L0:.3f} m"
            )
    print(
        "\nCamera tip: orbit with left-drag, zoom with scroll.\n"
        "Close the viewer window to exit.\n"
    )

    # ── Render loop ────────────────────────────────────────────────────────────
    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
