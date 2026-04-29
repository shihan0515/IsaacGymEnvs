#!/usr/bin/env python3
"""
Standalone IsaacGym visualization: Mug, Drill, Dumbbell placed evenly on a table.
Intended for thesis figure generation.

Usage (from IsaacGymEnvs/ directory):
    python show_objects_display.py
    python show_objects_display.py --pipeline cpu   (if GPU pipeline fails)
"""
import os
import sys

from isaacgym import gymapi, gymutil


def main():
    gym = gymapi.acquire_gym()
    args = gymutil.parse_arguments(description="Display three grasp objects on a table")

    # ------------------------------------------------------------------ #
    # Simulation parameters                                               #
    # ------------------------------------------------------------------ #
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.use_gpu_pipeline = False  # CPU is sufficient for static display

    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.005
        sim_params.physx.rest_offset = 0.0

    sim = gym.create_sim(
        args.compute_device_id, args.graphics_device_id,
        args.physics_engine, sim_params
    )
    if sim is None:
        print("ERROR: Failed to create sim")
        sys.exit(1)

    # Ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    # Viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("ERROR: Failed to create viewer")
        sys.exit(1)

    # Camera angle: elevated front-right view, framing all three objects
    cam_pos    = gymapi.Vec3(1.1, -0.85, 0.80)
    cam_target = gymapi.Vec3(0.45, 0.0, 0.34)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # ------------------------------------------------------------------ #
    # Asset root (same as DiabloGraspCustom3 task)                        #
    # ------------------------------------------------------------------ #
    asset_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets"
    )

    # ------------------------------------------------------------------ #
    # Table                                                               #
    # TABLE_W x TABLE_D surface, TABLE_T thick                           #
    # ------------------------------------------------------------------ #
    TABLE_W = 10    # x (depth seen from front)
    TABLE_D = 10    # y (width across the three objects)
    TABLE_T = 0.04    # thickness
    TABLE_CX = 0.45
    TABLE_CY = 0.00
    TABLE_CZ = 0.30   # centre of the box
    TABLE_SURFACE_Z = TABLE_CZ + TABLE_T / 2.0   # top face = 0.32 m

    table_opts = gymapi.AssetOptions()
    table_opts.fix_base_link = True
    table_asset = gym.create_box(sim, TABLE_W, TABLE_D, TABLE_T, table_opts)

    # ------------------------------------------------------------------ #
    # Object assets  (fix_base_link keeps them static – no physics drift) #
    # ------------------------------------------------------------------ #
    obj_opts = gymapi.AssetOptions()
    obj_opts.fix_base_link = True
    obj_opts.use_mesh_materials = True
    obj_opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    obj_opts.override_com = True
    obj_opts.override_inertia = True
    obj_opts.vhacd_enabled = False   # not needed for static display

    print(f"Asset root: {asset_root}")

    mug_file      = "urdf/ycb/025_mug/025_mug.urdf"
    drill_file    = "urdf/ycb/ycb_urdfs-main/ycb_assets/035_power_drill.urdf"
    dumbbell_file = "urdf/dumbbell.urdf"

    print(f"Loading mug      : {mug_file}")
    mug_asset      = gym.load_asset(sim, asset_root, mug_file, obj_opts)
    print(f"Loading drill    : {drill_file}")
    drill_asset    = gym.load_asset(sim, asset_root, drill_file, obj_opts)
    print(f"Loading dumbbell : {dumbbell_file}")
    dumbbell_asset = gym.load_asset(sim, asset_root, dumbbell_file, obj_opts)

    # ------------------------------------------------------------------ #
    # Single environment                                                  #
    # ------------------------------------------------------------------ #
    env_lower = gymapi.Vec3(-2.0, -2.0, 0.0)
    env_upper = gymapi.Vec3( 2.0,  2.0, 2.0)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # --- Table actor ---
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(TABLE_CX, TABLE_CY, TABLE_CZ)
    table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    table_actor = gym.create_actor(env, table_asset, table_pose, "table", 0, 1, 0)
    # White colour
    gym.set_rigid_body_color(
        env, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 1.0, 1.0)
    )

    # ------------------------------------------------------------------ #
    # Object placement: equally spaced along Y-axis                       #
    #                                                                     #
    #   Y = -0.25   Mug      (upright, half-height 0.0425 m)             #
    #   Y =  0.00   Drill    (upright, half-height 0.050  m)             #
    #   Y = +0.25   Dumbbell (upright along Z, half-height 0.064 m)       #
    # ------------------------------------------------------------------ #
    SPACING = 0.25  # metres between object centres

    # --- Mug (upright) ---
    # The 025_mug URDF is oriented with the mug standing along Z.
    # half-height = 0.0425 m (from task config objectHalfHeight)
    MUG_HALF_H = 0.0425
    mug_pose = gymapi.Transform()
    mug_pose.p = gymapi.Vec3(TABLE_CX, -SPACING, TABLE_SURFACE_Z + MUG_HALF_H)
    mug_pose.r = gymapi.Quat(0.0, 0.0, 0.7071068, 0.7071068)  # 90° about Z
    gym.create_actor(env, mug_asset, mug_pose, "mug", 0, 2, 0)

    # --- Drill (upright) ---
    # The 035_power_drill URDF already has rpy="π/2 0 0" baked into its visual,
    # so identity actor rotation produces the standing pose.
    # half-height = 0.050 m (from task config objectHalfHeight)
    DRILL_HALF_H = 0.050
    drill_pose = gymapi.Transform()
    drill_pose.p = gymapi.Vec3(TABLE_CX, 0.0, TABLE_SURFACE_Z + DRILL_HALF_H)
    drill_pose.r = gymapi.Quat(0.0, 0.0, 0.7071068, 0.7071068)  # 90° about Z
    gym.create_actor(env, drill_asset, drill_pose, "drill", 0, 3, 0)

    # --- Dumbbell (upright, handle along Z) ---
    # The dumbbell URDF has its handle cylinder along Z.
    # Bottom face (right_weight bottom) is at Z = -0.052 - 0.012 = -0.064 m from origin.
    # Place origin at TABLE_SURFACE_Z + 0.064 so the disc rests flush on the table.
    DUMBBELL_HALF_H = 0.064
    dumbbell_pose = gymapi.Transform()
    dumbbell_pose.p = gymapi.Vec3(TABLE_CX, SPACING, TABLE_SURFACE_Z + DUMBBELL_HALF_H)
    dumbbell_pose.r = gymapi.Quat(0.0, 0.0, 0.7071068, 0.7071068)  # 90° about Z
    gym.create_actor(env, dumbbell_asset, dumbbell_pose, "dumbbell", 0, 4, 0)

    # ------------------------------------------------------------------ #
    # Run simulation loop                                                 #
    # ------------------------------------------------------------------ #
    gym.prepare_sim(sim)

    print("\n=== Object display ready ===")
    print(f"  Table surface Z = {TABLE_SURFACE_Z:.3f} m")
    print(f"  Mug      at Y = {-SPACING:+.2f} m  (upright)")
    print(f"  Drill    at Y = {0.0:+.2f} m  (upright)")
    print(f"  Dumbbell at Y = {+SPACING:+.2f} m  (upright)")
    print("Close the viewer window to exit.\n")

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
