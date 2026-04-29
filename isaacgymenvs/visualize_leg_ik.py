import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Polygon # Import Polygon

# --- 物理參數 (與 diablo_graspcustom.py 中修正後的一致) ---
R = 0.08  # 輪子半徑 (m)
L1 = 0.14  # 大腿長度 (m)
L2 = 0.14  # 小腿長度 (m)
BASE_LENGTH = 0.3 # 假設的 base 長度
BASE_HEIGHT = 0.1 # 假設的 base 高度

def calculate_leg_kinematics(h_target, p_target):
    """
    根據目標高度和俯仰角，計算腿部的完整運動學狀態。
    返回臀部、膝蓋和輪子中心的三個點的座標。
    """
    # 1. 計算所需的有效腿長 L0
    L0 = (h_target - R) / np.cos(p_target)
    L0 = np.clip(L0, 0.01, L1 + L2) # 限制L0在物理極限內

    # 2. 計算各點在機器人參考系下的座標 (臀部為原點(0,0))
    # 身體傾斜 p_target, 腿部連桿相對於垂直向下方向的角度為 -p_target
    # 輪子中心相對於臀部的座標
    wheel_center_x = L0 * np.sin(-p_target)
    wheel_center_y = -L0 * np.cos(-p_target)

    # 3. 使用兩圓相交法計算膝蓋位置
    # (參考: https://stackoverflow.com/questions/5580318/how-to-find-the-intersection-of-two-circles)
    d = L0 # 臀部到輪子中心的距離
    
    # 檢查是否可解
    if d > L1 + L2 or d < abs(L1 - L2):
        # 如果無解 (腿伸太直或太縮)，則將膝蓋放在連線上
        knee_x = wheel_center_x * (L1 / d)
        knee_y = wheel_center_y * (L1 / d)
    else:
        a = (L1**2 - L2**2 + d**2) / (2 * d)
        h = np.sqrt(max(0, L1**2 - a**2))
        
        # 中點座標
        mid_x = 0 + a * (wheel_center_x - 0) / d
        mid_y = 0 + a * (wheel_center_y - 0) / d

        # 選擇 "向後彎曲" 的解
        knee_x = mid_x + h * (wheel_center_y - 0) / d
        knee_y = mid_y - h * (wheel_center_x - 0) / d

    # 4. 將座標轉換到世界參考系 (地面y=0, 臀部x=0)
    hip_pos = np.array([0, h_target])
    knee_pos = np.array([knee_x, knee_y]) + hip_pos
    wheel_center_pos = np.array([wheel_center_x, wheel_center_y]) + hip_pos
    
    return hip_pos, knee_pos, wheel_center_pos

# --- 繪圖設定 ---
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(left=0.1, bottom=0.35) # 調整子圖位置，為滑桿留出空間

# 初始值
initial_h = 0.29
initial_p_rad = 0.0

# 繪製初始圖形
hip, knee, wheel = calculate_leg_kinematics(initial_h, initial_p_rad)
thigh_line, = ax.plot([hip[0], knee[0]], [hip[1], knee[1]], 'r-', lw=4, label='thigh (L1)')
shank_line, = ax.plot([knee[0], wheel[0]], [knee[1], wheel[1]], 'b-', lw=4, label='shank (L2)')
hip_dot, = ax.plot(hip[0], hip[1], 'rs', markersize=8, label='hip')
knee_dot, = ax.plot(knee[0], knee[1], 'ms', markersize=8, label='knee')
wheel_circle = plt.Circle(wheel, R, color='g', fill=False, lw=2, label='wheel')
ax.add_patch(wheel_circle)
ground_line = ax.axhline(y=0, color='black', linestyle='--', label='ground')

# --- 新增: 繪製可傾斜的 base (Polygon) ---
# 初始 base 繪製 (水平)
base_half_len = BASE_LENGTH / 2
base_half_height = BASE_HEIGHT / 2
base_corners_unrotated = np.array([
    [-base_half_len, -base_half_height],
    [ base_half_len, -base_half_height],
    [ base_half_len,  base_half_height],
    [-base_half_len,  base_half_height]
])
# 初始旋轉矩陣 (0度)
rot_matrix_initial = np.array([[np.cos(initial_p_rad), -np.sin(initial_p_rad)],
                               [np.sin(initial_p_rad),  np.cos(initial_p_rad)]])
rotated_corners_initial = (rot_matrix_initial @ base_corners_unrotated.T).T
base_polygon = Polygon(rotated_corners_initial + hip, closed=True, color='purple', alpha=0.7, label='base')
ax.add_patch(base_polygon)
# ---

height_text = ax.text(0.35, initial_h / 2, f'height: {initial_h:.2f} m', fontsize=12)
pitch_text = ax.text(0.35, initial_h / 2 - 0.03, f'pitch: {np.degrees(initial_p_rad):.1f}°', fontsize=12)


# 設定座標軸
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-0.4, 0.4)
ax.set_ylim(-0.1, 0.5)
ax.set_title('Diablo IK Leg Visualization with Base Tilt')
ax.set_xlabel('X axis (m)')
ax.set_ylabel('Y axis (m)')
ax.grid(True)
ax.legend()

# --- 滑桿設定 ---
ax_h = plt.axes([0.25, 0.15, 0.65, 0.03]) # 高度滑桿位置
ax_p = plt.axes([0.25, 0.1, 0.65, 0.03])  # 傾斜滑桿位置

h_slider = Slider(
    ax=ax_h,
    label='Height (m)',
    valmin=0.23,
    valmax=0.35,
    valinit=initial_h,
)

p_slider = Slider(
    ax=ax_p,
    label='Pitch (deg)',
    valmin=-22, # approx -0.4 rad
    valmax=22,  # approx +0.4 rad
    valinit=np.degrees(initial_p_rad),
)

# --- 更新函數 ---
def update(val):
    h = h_slider.val
    p_rad = np.radians(p_slider.val)
    
    hip, knee, wheel = calculate_leg_kinematics(h, p_rad)
    
    # 更新腿部
    thigh_line.set_data([hip[0], knee[0]], [hip[1], knee[1]])
    shank_line.set_data([knee[0], wheel[0]], [knee[1], wheel[1]])
    hip_dot.set_data(hip[0], hip[1])
    knee_dot.set_data(knee[0], knee[1])
    wheel_circle.center = wheel
    
    # --- 新增: 更新 base 的傾斜 (Polygon) ---
    rot_matrix = np.array([[np.cos(p_rad), -np.sin(p_rad)],
                           [np.sin(p_rad),  np.cos(p_rad)]])
    rotated_corners = (rot_matrix @ base_corners_unrotated.T).T
    base_polygon.set_xy(rotated_corners + hip)
    # ---

    # 更新文字
    height_text.set_position((0.35, h / 2))
    height_text.set_text(f'height: {h:.2f} m')
    pitch_text.set_position((0.35, h / 2 - 0.03))
    pitch_text.set_text(f'pitch: {np.degrees(p_rad):.1f}°')

    fig.canvas.draw_idle()

# 綁定滑桿與更新函數
h_slider.on_changed(update)
p_slider.on_changed(update)

plt.show()