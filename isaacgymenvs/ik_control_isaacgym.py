#!/usr/bin/env python3
"""
Interactive Diablo IK control – IsaacGym + matplotlib sliders.

IK identical to diablo_graspcustom3.py reset_idx:
  L0     = (h − R) / cos(pitch)          virtual leg length
  knee   = π − acos((l1²+l2²−L0²) / (2·l1·l2))
  hip    = −pitch − acos(L0 / (2·l1))

VMC parameters (DDTRobot convention, computed from IK output):
  L0     : virtual leg length  (m)
  theta0 : leg angle from vertical ≈ −pitch  (rad)

Height range matches training: 0.20 – 0.35 m.
  Theoretical max is L1 + L2 + R_wheel = 0.36 m, but at that singularity
  knee_dof → 0 conflicts with the URDF joint-frame offset and looks wrong.

Architecture:
  Main thread  : IsaacGym 3-D viewer – updates robot in real-time
  Daemon thread: matplotlib window – 2D side-view + 3 sliders

Sliders:
  Height (m)  : 0.20 – 0.35
  Pitch  (°)  : −20  – +20
  Yaw    (°)  : −25  – +25

Usage (from isaacgymenvs/ directory):
    python ik_control_isaacgym.py
    python ik_control_isaacgym.py --pipeline cpu
"""

import math
import os
import sys
import threading

import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
import torch


# ── Physical parameters ────────────────────────────────────────────────────────

R_WHEEL = 0.08          # wheel radius (m) – from diablo_graspcustom3.py
L1, L2  = 0.14, 0.14   # thigh / shank length (m)

# Joint names in our URDF
LEG_NAMES = [
    "left_fake_hip_joint",  "left_fake_knee_joint",
    "right_fake_hip_joint", "right_fake_knee_joint",
]
ARM_NAMES   = ["r_sho_pitch", "r_sho_roll", "r_el", "r_wrist"]
ARM_DEFAULT = np.radians([0, -60, -80, 0])


# ── Thread-safe shared slider state ───────────────────────────────────────────

class _Shared:
    def __init__(self):
        self._h = 0.29
        self._p = 0.0
        self._y = 0.0
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            return self._h, self._p, self._y

    def set(self, h, p, y):
        with self._lock:
            self._h, self._p, self._y = h, p, y

_shared = _Shared()


# ── IK / FK ───────────────────────────────────────────────────────────────────

def compute_leg_ik(h: float, p: float):
    """
    IK: (height, pitch) → (hip_dof, knee_dof, L0).
    Identical to reset_idx in diablo_graspcustom3.py.
    """
    L0 = max(0.01, min((h - R_WHEEL) / math.cos(p), L1 + L2))
    cos_alpha = max(-1.0, min(1.0, (L1**2 + L2**2 - L0**2) / (2 * L1 * L2)))
    knee_dof  = math.pi - math.acos(cos_alpha)
    cos_beta  = max(-1.0, min(1.0, L0 / (2 * L1)))
    hip_dof   = -p - math.acos(cos_beta)
    return hip_dof, knee_dof, L0


def _ik_2d_positions(h: float, p: float):
    """
    2D side-view joint positions using two-circle intersection (backward-bending knee).
    Coordinate frame: +X forward, +Y upward, hip at (0, h).
    """
    L0 = max(0.01, min((h - R_WHEEL) / math.cos(p), L1 + L2))

    # Wheel center relative to hip (leg hangs at angle -p from vertical)
    wx = L0 * math.sin(-p)
    wy = -L0 * math.cos(p)

    d = L0
    if d > L1 + L2 or d < abs(L1 - L2):
        kx = wx * (L1 / d)
        ky = wy * (L1 / d)
    else:
        a  = (L1**2 - L2**2 + d**2) / (2 * d)
        hh = math.sqrt(max(0.0, L1**2 - a**2))
        mx = a * wx / d
        my = a * wy / d
        # Backward-bending solution (knee bends behind the leg axis)
        kx = mx + hh * wy / d
        ky = my - hh * wx / d

    hip_2d   = np.array([0.0, h])
    knee_2d  = np.array([kx, ky]) + hip_2d
    wheel_2d = np.array([wx, wy]) + hip_2d
    return hip_2d, knee_2d, wheel_2d


def make_base_quat(p: float, y: float):
    """quat_from_euler_xyz(roll=0, pitch=p, yaw=y) → (qx, qy, qz, qw)."""
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    return -sy * sp, cy * sp, sy * cp, cy * cp


# ── matplotlib slider window (daemon thread) ──────────────────────────────────

def _slider_thread_fn():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        from matplotlib.patches import Circle, Polygon as MplPolygon
    except ImportError:
        print("[slider] matplotlib not found – sliders disabled.")
        return

    BASE_LEN, BASE_HGT = 0.30, 0.10
    bl, bh = BASE_LEN / 2, BASE_HGT / 2
    corners0 = np.array([[-bl, -bh], [bl, -bh], [bl, bh], [-bl, bh]])

    fig = plt.figure(figsize=(8, 9))
    try:
        fig.canvas.manager.set_window_title("Diablo VMC / IK Controller")
    except Exception:
        pass
    plt.subplots_adjust(left=0.10, bottom=0.33)

    # ── 2D side-view axes ──────────────────────────────────────────────────────
    ax = fig.add_axes([0.08, 0.35, 0.88, 0.57])

    h0, p0 = 0.29, 0.0
    hip0, knee0, wh0 = _ik_2d_positions(h0, p0)
    _, _, L0_0 = compute_leg_ik(h0, p0)

    thigh_ln, = ax.plot([hip0[0], knee0[0]], [hip0[1], knee0[1]],
                        'r-', lw=4, label='Thigh  L1 = 0.14 m')
    shank_ln, = ax.plot([knee0[0], wh0[0]],  [knee0[1], wh0[1]],
                        'b-', lw=4, label='Shank  L2 = 0.14 m')
    vleg_ln,  = ax.plot([hip0[0], wh0[0]],   [hip0[1], wh0[1]],
                        'k--', lw=1.2, alpha=0.5, label='Virtual leg L0')
    hip_pt,   = ax.plot(hip0[0],  hip0[1],  'rs', ms=9,  label='Hip  joint')
    knee_pt,  = ax.plot(knee0[0], knee0[1], 'ms', ms=9,  label='Knee joint')
    wheel_c   = Circle(tuple(wh0), R_WHEEL, color='g', fill=False, lw=2,
                       label=f'Wheel  R={R_WHEEL} m')
    ax.add_patch(wheel_c)

    Rm0 = np.array([[math.cos(p0), -math.sin(p0)],
                    [math.sin(p0),  math.cos(p0)]])
    base_poly = MplPolygon((Rm0 @ corners0.T).T + hip0, closed=True,
                           color='purple', alpha=0.65, label='Body')
    ax.add_patch(base_poly)

    ax.axhline(y=0, color='k', ls='--', lw=1)
    ax.axhline(y=R_WHEEL, color='g', ls=':', lw=0.8, alpha=0.5)

    # Info text (two rows)
    info1 = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                    fontsize=9.5, color='navy', va='top',
                    fontfamily='monospace')
    info2 = ax.text(0.02, 0.88, '', transform=ax.transAxes,
                    fontsize=9.5, color='darkred', va='top',
                    fontfamily='monospace')

    ax.set_aspect('equal', 'box')
    ax.set_xlim(-0.35, 0.35)
    ax.set_ylim(-0.12, 0.52)
    ax.set_title(
        'Diablo IK – 2D Side View\n'
        'knee bends backward  |  +X → forward  |  +Y → up',
        fontsize=10
    )
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=7.5, ncol=2)

    # ── Sliders ────────────────────────────────────────────────────────────────
    ax_h = fig.add_axes([0.18, 0.23, 0.72, 0.04])
    ax_p = fig.add_axes([0.18, 0.15, 0.72, 0.04])
    ax_y = fig.add_axes([0.18, 0.07, 0.72, 0.04])

    sl_h = Slider(ax_h, 'Height (m)', 0.20, 0.35, valinit=0.29,
                  valstep=0.002, color='steelblue')
    sl_p = Slider(ax_p, 'Pitch  (°)', -20.0, 20.0, valinit=0.0,
                  valstep=0.5,   color='tomato')
    sl_y = Slider(ax_y, 'Yaw    (°)', -25.0, 25.0, valinit=0.0,
                  valstep=0.5,   color='seagreen')

    def _update(_val):
        h = sl_h.val
        p = math.radians(sl_p.val)
        y = math.radians(sl_y.val)
        _shared.set(h, p, y)

        # Recompute IK
        hip_dof, knee_dof, L0 = compute_leg_ik(h, p)
        theta0 = -p  # VMC leg angle from vertical ≈ -pitch

        # Update 2D plot
        hip, knee, wheel = _ik_2d_positions(h, p)
        thigh_ln.set_data([hip[0], knee[0]], [hip[1], knee[1]])
        shank_ln.set_data([knee[0], wheel[0]], [knee[1], wheel[1]])
        vleg_ln.set_data( [hip[0], wheel[0]], [hip[1], wheel[1]])
        hip_pt.set_data([hip[0]],   [hip[1]])
        knee_pt.set_data([knee[0]], [knee[1]])
        wheel_c.center = tuple(wheel)

        Rm = np.array([[math.cos(p), -math.sin(p)],
                       [math.sin(p),  math.cos(p)]])
        base_poly.set_xy((Rm @ corners0.T).T + hip)

        # ── Info line 1: slider values + DOF angles ──
        info1.set_text(
            f'h = {h:.3f} m    pitch = {sl_p.val:+.1f}°    yaw = {sl_y.val:+.1f}°\n'
            f'hip_dof = {math.degrees(hip_dof):+.1f}°    '
            f'knee_dof = {math.degrees(knee_dof):+.1f}°    '
            f'L0 = {L0:.3f} m'
        )

        # ── Info line 2: VMC parameters ──
        info2.set_text(
            f'L0 = {L0:.3f} m    '
            f'θ0 = {math.degrees(theta0):+.1f}°  (≈ −pitch)'
        )

        fig.canvas.draw_idle()

    sl_h.on_changed(_update)
    sl_p.on_changed(_update)
    sl_y.on_changed(_update)
    _update(None)

    plt.show()


# ── IsaacGym main ─────────────────────────────────────────────────────────────

def main():
    t = threading.Thread(target=_slider_thread_fn, daemon=True)
    t.start()

    gym  = gymapi.acquire_gym()
    args = gymutil.parse_arguments(
        description="Interactive Diablo IK/VMC control (IsaacGym + sliders)"
    )

    sp = gymapi.SimParams()
    sp.up_axis  = gymapi.UP_AXIS_Z
    sp.gravity  = gymapi.Vec3(0.0, 0.0, -9.81)
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

    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane)

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("ERROR: create_viewer failed"); sys.exit(1)

    gym.viewer_camera_look_at(
        viewer, None,
        gymapi.Vec3(0.1, -1.5, 0.8),
        gymapi.Vec3(-0.05, 0.0, 0.23),
    )

    # ── Asset ──────────────────────────────────────────────────────────────────
    asset_root  = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../assets"
    )
    diablo_file = (
        "urdf/diab/Diablo_ger/Part_gripper_col_rev/URDF/diablo_erc_visualize.urdf"
    )

    ao = gymapi.AssetOptions()
    ao.fix_base_link           = True
    ao.collapse_fixed_joints   = False
    ao.disable_gravity         = True
    ao.default_dof_drive_mode  = gymapi.DOF_MODE_POS
    ao.use_mesh_materials      = False
    ao.flip_visual_attachments = False
    ao.armature                = 0.01

    print(f"Loading {diablo_file} ...")
    asset    = gym.load_asset(sim, asset_root, diablo_file, ao)
    num_dofs = gym.get_asset_dof_count(asset)
    dof_dict = gym.get_asset_dof_dict(asset)

    leg_ids = [dof_dict[n] for n in LEG_NAMES]
    arm_ids = [dof_dict[n] for n in ARM_NAMES]
    print(f"Leg DOF IDs : {dict(zip(LEG_NAMES, leg_ids))}")
    print(f"Total DOFs  : {num_dofs}")

    dof_props = gym.get_asset_dof_properties(asset)
    for i in range(num_dofs):
        dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
        dof_props["stiffness"][i] = 10000.0
        dof_props["damping"][i]   = 1000.0

    # ── Table asset (identical to diablo_graspcustom3.py) ─────────────────────
    table_opts = gymapi.AssetOptions()
    table_opts.fix_base_link = True
    table_asset = gym.create_box(sim, 0.3, 0.5, 0.01, table_opts)

    env_lower = gymapi.Vec3(-1.5, -1.5, 0.0)
    env_upper = gymapi.Vec3( 1.5,  1.5, 2.5)
    env   = gym.create_env(sim, env_lower, env_upper, 1)

    init_pose   = gymapi.Transform()
    init_pose.p = gymapi.Vec3(0.0, 0.0, 0.29)
    init_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actor = gym.create_actor(env, asset, init_pose, "diablo", 0, -1, 0)
    gym.set_actor_dof_properties(env, actor, dof_props)

    table_pose   = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.25, 0.0, 0.35)
    table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    gym.create_actor(env, table_asset, table_pose, "table", 0, 1, 0)

    # ── Tensor setup ───────────────────────────────────────────────────────────
    gym.prepare_sim(sim)

    root_state = gymtorch.wrap_tensor(
        gym.acquire_actor_root_state_tensor(sim)
    )
    dof_state = gymtorch.wrap_tensor(
        gym.acquire_dof_state_tensor(sim)
    )

    target = torch.zeros(num_dofs, dtype=torch.float32)
    dof_state[:, :] = 0.0
    for j, aid in enumerate(arm_ids):
        v = float(ARM_DEFAULT[j])
        dof_state[aid, 0] = v
        target[aid]        = v

    actor_indices = torch.tensor([0], dtype=torch.int32)

    h0, p0, _ = _shared.get()
    hip0, knee0, _ = compute_leg_ik(h0, p0)
    for j, lid in enumerate(leg_ids):
        v = float(hip0) if j % 2 == 0 else float(knee0)
        dof_state[lid, 0] = v
        target[lid]        = v

    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(target))

    print("\n=== Interactive Diablo IK Control ===")
    print("  Sliders → Height 0.20–0.35 m | Pitch ±20° | Yaw ±25°")
    print("  matplotlib window shows: DOF angles + VMC params (L0, θ0)")
    print("  Close the IsaacGym viewer to exit.\n")

    # ── Main render loop ───────────────────────────────────────────────────────
    while not gym.query_viewer_has_closed(viewer):
        h, p, y = _shared.get()
        hip, knee, L0 = compute_leg_ik(h, p)

        bx = L0 * math.sin(p) * math.cos(y)
        by = L0 * math.sin(p) * math.sin(y)
        qx, qy, qz, qw = make_base_quat(p, y)

        root_state[0, 0] = bx
        root_state[0, 1] = by
        root_state[0, 2] = h
        root_state[0, 3] = qx
        root_state[0, 4] = qy
        root_state[0, 5] = qz
        root_state[0, 6] = qw
        root_state[0, 7:13] = 0.0

        gym.set_actor_root_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(root_state),
            gymtorch.unwrap_tensor(actor_indices), 1
        )

        for j, lid in enumerate(leg_ids):
            v = float(hip) if j % 2 == 0 else float(knee)
            dof_state[lid, 0] = v
            dof_state[lid, 1] = 0.0
            target[lid]        = v

        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(target))

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
