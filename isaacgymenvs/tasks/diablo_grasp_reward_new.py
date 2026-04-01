@torch.jit.script
def compute_diablo_reward(
    reset_buf, progress_buf, actions,
    object_z, initial_object_z,
    eef_pos, handle_pos, eef_rot, handle_rot,
    mug_rot,
    gripper_dof_pos, gripper_lower_limits, gripper_upper_limits,
    dist_reward_scale, rot_reward_scale, lift_reward_scale, grasp_reward_scale,
    orientation_reward_scale, action_penalty_scale,
    max_episode_length,
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

    # --- Stage 1: Reaching and Alignment Rewards ---
    d = torch.norm(eef_pos - handle_pos, p=2, dim=-1)
    # 距離獎勵，提供平滑梯度
    dist_reward = torch.exp(-2.0 * d) 
    # 接近加成
    dist_reward = torch.where(d <= 0.15, dist_reward + 2.0 * torch.exp(-10.0 * d), dist_reward)

    # 旋轉對齊 (Rotation alignment)
    # Gripper axes (in local frame)
    # -Z 是手掌前方, X 是手腕向上方向
    gripper_forward_axis = torch.tensor([0.0, 0.0, -1.0], device=eef_pos.device).repeat(eef_pos.shape[0], 1)
    gripper_up_axis = torch.tensor([1.0, 0.0, 0.0], device=eef_pos.device).repeat(eef_pos.shape[0], 1)
    
    # Handle axes (relative to handle's own orientation)
    # 既然杯子初始旋轉了 180 度，本地 +X 會指向機器人，
    # 我們希望夾爪前方 (-Z) 對齊杯柄的本地 -X 方向（即世界空間的 +X），達成面對面抓取。
    handle_forward_target = torch.tensor([-1.0, 0.0, 0.0], device=eef_pos.device).repeat(eef_pos.shape[0], 1)
    handle_up_axis = torch.tensor([0.0, 0.0, 1.0], device=eef_pos.device).repeat(eef_pos.shape[0], 1)

    axis1 = tf_vector(eef_rot, gripper_forward_axis) # Gripper forward in world
    axis2 = tf_vector(handle_rot, handle_forward_target) # Handle approach direction in world
    axis3 = tf_vector(eef_rot, gripper_up_axis) # Gripper up (X-axis) in world
    axis4 = tf_vector(handle_rot, handle_up_axis) # Handle up in world

    # 旋轉對齊計算：forward 向量一致
    dot1 = torch.clamp(torch.sum(axis1 * axis2, dim=-1), min=0.0)
    # 這裡我們不強制 up 向量完全一致（dot2），而是將「X軸朝上」作為獨立約束
    rot_alignment = dot1 
    rot_reward = rot_alignment

    # --- 新增：EEF 姿態約束 (Grip-site X 軸朝上) ---
    world_z_axis = torch.tensor([0.0, 0.0, 1.0], device=eef_pos.device).repeat(eef_pos.shape[0], 1)
    # axis3 就是夾爪的局部 X 軸在世界空間的方向
    upright_alignment = torch.clamp(torch.sum(axis3 * world_z_axis, dim=-1), min=0.0)
    orientation_reward = upright_alignment
   
    # 距離與對齊連動，確保 AI 在正確的方向接近，且姿勢正確
    # 我們讓姿態對齊對接近獎勵有一定的權重，這會強制 AI 在接近杯柄時就維持 X 軸向上
    reach_reward = dist_reward * (0.33 + 0.33 * rot_alignment + 0.34 * upright_alignment)
   
    # --- Stage 2: Grasping Logic ---
    # 計算夾爪狀態
    gripper_range = gripper_upper_limits - gripper_lower_limits
    gripper_relative_pos = (gripper_dof_pos - gripper_lower_limits) / (gripper_range + 1e-6)
    # 物理上閉合判定 (放寬到 0.5，避免因物體厚度導致無法達成獎勵)
    gripper_physically_closed = torch.mean(gripper_relative_pos, dim=-1) > 0.5
    
    # 動作意圖：夾爪閉合指令 (Action 13)
    gripper_close_action = (actions[:, 13] >= 0.0)
    
    # 抓取判定區域
    is_near_handle = (d < 0.06)
    is_in_grasp_zone = (d < 0.035) & (rot_alignment > 0.6) 
    
    # 嘗試合攏獎勵：在手柄附近且有閉合動作
    grasp_attempt_reward = torch.where(is_near_handle & gripper_close_action, torch.ones_like(dist_reward) * 2.0, torch.zeros_like(dist_reward))
    
    # --- Stage 3: Lifting Reward ---
    lifted_height = torch.clamp(object_z - initial_object_z, min=0.0)
    # 核心修正：只要杯子被舉起，不論夾爪狀態如何，都給予基礎舉起獎勵 (避免梯度陷阱)
    is_actually_lifting = (lifted_height > 0.01)
    is_grasping_success = is_in_grasp_zone & gripper_physically_closed
    
    # 基礎舉起獎勵
    lift_reward = torch.where(is_actually_lifting, 
                              20.0 + lifted_height * 200.0, 
                              torch.zeros_like(dist_reward))
    
    # 強力引導：如果是透過「正確抓取」而舉起的，獎勵加倍 (2.0x)
    lift_reward = torch.where(is_actually_lifting & is_grasping_success,
                              lift_reward * 2.0, 
                              lift_reward)
    
    # 如果抓取動作正確 (即使還沒舉起)，給予額外的鼓勵獎勵
    grasp_reward = torch.where(is_grasping_success, torch.ones_like(dist_reward) * 5.0, torch.zeros_like(dist_reward))

    # --- Stage 4: Penalties ---
    # 防止空手舉起：如果手向上移動超過 4cm 但杯子沒動，給予懲罰
    eef_higher_than_handle = (eef_pos[:, 2] > handle_pos[:, 2] + 0.04)
    lift_penalty = torch.where(eef_higher_than_handle & ~is_actually_lifting, 
                               -10.0 * (eef_pos[:, 2] - handle_pos[:, 2]), 
                               torch.zeros_like(dist_reward))
    
    # 如果手臂太高且沒抓到東西，大幅削減接近獎勵
    reach_reward = torch.where(eef_higher_than_handle & ~is_actually_lifting, reach_reward * 0.1, reach_reward)

    # 動作懲罰
    action_penalty = torch.sum(actions ** 2, dim=-1)
    actual_action_penalty = action_penalty_scale * action_penalty

    # --- 總獎勵合成 (使用傳入的 Scale) ---
    rewards = dist_reward_scale * reach_reward 
            + rot_reward_scale * rot_reward 
            + orientation_reward_scale * orientation_reward 
            + grasp_reward_scale * (grasp_attempt_reward + grasp_reward) 
            + lift_reward_scale * lift_reward 
            + lift_penalty 
            - actual_action_penalty
            

    # --- 智能重置 ---
    # 掉落重置
    reset_buf = torch.where(object_z < 0.1, torch.ones_like(reset_buf), reset_buf)
    # 超時重置
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    # 成功重置：成功舉起到 20cm 以上
    reset_buf = torch.where(lifted_height > 0.2, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf
