# Diablo II 抓取任務：參數與公式

本文檔整理了 `diablo2.py` 任務中用於訓練機器人抓取馬克杯的核心參數和數學公式，可作為研究論文中方法論章節的參考。

## 1. 環境設定 (Environment Setup)

- **模擬器**: NVIDIA Isaac Gym
- **物理引擎**: NVIDIA PhysX
- **核心資產 (Assets)**:
    - **機器人**: Diablo（雙臂人形機器人，本任務僅使用其右臂和頭部固定）
    - **目標物體**: YCB Dataset 中的馬克杯 (`025_mug.urdf`)
    - **場景**: 一張桌子 (`table`) 和一個放置馬克杯的較高平台 (`table_stand`)
- **物理參數**:
    - **重力**: `g = -9.81 m/s²` (沿 Z 軸負方向)

## 2. 觀測空間 (Observation Space)

代理在每個時間步接收一個 78 維的狀態向量作為觀測值，其組成如下：

- **回合進度 (Episode Progress)** `(1 維)`: 當前回合步數與最大步數的比例。
- **關節位置 (DOF Positions)** `(28 維)`: 機器人 28 個關節的標準化位置。
  - 標準化公式: $p_{norm} = 2 \cdot \frac{p_{raw} - p_{lower}}{p_{upper} - p_{lower}} - 1$
- **關節速度 (DOF Velocities)** `(28 維)`: 機器人 28 個關節的縮放後速度。
- **馬克杯位置 (Mug Position)** `(3 維)`: 馬克杯在世界座標系中的 `(x, y, z)` 位置。
- **馬克杯姿態 (Mug Rotation)** `(4 維)`: 馬克杯在世界座標系中的四元數 `(x, y, z, w)`。
- **上一時刻動作 (Previous Action)** `(14 維)`: 代理在上一時間步執行的動作。

## 3. 動作空間 (Action Space)

代理的動作空間為 14 維的連續向量，控制方式為關節空間 (`joint`) 控制。

- **右臂關節 (Right Arm Joints)** `(13 維)`: 控制右手 13 個關節的目標位置。
- **夾爪控制 (Gripper Control)** `(1 維)`: 一個開關量，用於控制三指夾爪的開合。
  - `action[13] >= 0`: 命令夾爪閉合至其關節上限。
  - `action[13] < 0`: 命令夾爪張開至其關節下限。

**動作映射**: 原始動作向量 $\mathbf{a}_{raw}$ 會經過縮放 (`action_scale`) 並疊加到預設姿態 $\mathbf{p}_{default}$ 上，以計算最終的目標關節位置 $\mathbf{p}_{target}$。
$$ \mathbf{p}_{target} = \text{action\_scale} \cdot \mathbf{a}_{raw} + \mathbf{p}_{default} $$

## 4. 獎勵函數 (Reward Function)

總獎勵 $R_{total}$ 是多個子獎勵的加權和，旨在分階段引導代理完成任務。

$$ R_{total} = w_{dist}R_{dist} + w_{rot}R_{rot} + R_{grasp} + w_{lift}R_{lift} + w_{orient}R_{orient} - w_{action}P_{action} $$

---

#### 4.1. 接近獎勵 ($R_{dist}$)

鼓勵末端執行器 $\mathbf{p}_{eef}$ 靠近目標握把 $\mathbf{p}_{handle}$。

- **距離**: $d = ||\mathbf{p}_{eef} - \mathbf{p}_{handle}||_2$
- **獎勵公式**:
  $$ R_{dist} = \begin{cases} 2 \cdot \left(\frac{1}{1 + d^2}\right)^2 & \text{if } d \le 0.05 \\ \left(\frac{1}{1 + d^2}\right)^2 & \text{otherwise} \end{cases} $$

---

#### 4.2. 姿態對齊獎勵 ($R_{rot}$)

鼓勵夾爪以正確姿態對準握把。

- **夾爪姿態軸**: 前向 $\mathbf{v}_{fwd\_grip}$ (Z-), 向上 $\mathbf{v}_{up\_grip}$ (X+)
- **握把姿態軸**: 向內 $\mathbf{v}_{in\_handle}$ (X+), 向上 $\mathbf{v}_{up\_handle}$ (Z+)
- **世界座標下的姿態軸**:
  - $\mathbf{a}_1 = \text{quat\_apply}(\mathbf{q}_{eef}, \mathbf{v}_{fwd\_grip})$
  - $\mathbf{a}_2 = \text{quat\_apply}(\mathbf{q}_{handle}, \mathbf{v}_{in\_handle})$
  - $\mathbf{a}_3 = \text{quat\_apply}(\mathbf{q}_{eef}, \mathbf{v}_{up\_grip})$
  - $\mathbf{a}_4 = \text{quat\_apply}(\mathbf{q}_{handle}, \mathbf{v}_{up\_handle})$
- **點積計算**:
  - $dot_1 = \mathbf{a}_1 \cdot \mathbf{a}_2$ (期望值 $\approx -1$)
  - $dot_2 = \mathbf{a}_3 \cdot \mathbf{a}_4$ (期望值 $\approx +1$)
- **獎勵公式**:
  $$ R_{rot} = 0.5 \cdot (\text{sgn}(dot_1) \cdot dot_1^2 + \text{sgn}(dot_2) \cdot dot_2^2) $$

---

#### 4.3. 嘗試抓取獎勵 ($R_{grasp}$)

在正確的位置和姿態下嘗試閉合夾爪時給予的稀疏獎勵。

- **抓取條件**:
  - `is_oriented` = $(dot_1 < -0.9) \land (dot_2 > 0.9)$
  - `is_close` = $(d < 0.025) \land \text{is\_oriented}$
  - `is_closing` = $(a_{gripper} \ge 0.0)$
- **獎勵公式**:
  $$ R_{grasp} = \begin{cases} 0.5 & \text{if } \text{is\_close} \land \text{is\_closing} \\ 0 & \text{otherwise} \end{cases} $$

---

#### 4.4. 提起物體獎勵 ($R_{lift}$)

在抓取物體後將其向上提起時給予的獎勵。

- **提起高度**: $h_{lift} = p_{mug, z} - p_{mug, z\_initial}$
- **抓取狀態**: `is_grasping` = `is_close` $\land$ `is_closing`
- **獎勵公式**:
  $$ R_{lift} = \begin{cases} 2.5 & \text{if is\_grasping} \land (h_{lift} > 0.05) \\ 1.0 & \text{if is\_grasping} \land (h_{lift} > 0.01) \\ 0 & \text{otherwise} \end{cases} $$

---

#### 4.5. 保持直立獎勵 ($R_{orient}$)

提起馬克杯後保持其直立。

- **姿態計算**:
  - $\mathbf{v}_{up\_world} = [0, 0, 1]^T$
  - $\mathbf{v}_{up\_mug} = \text{quat\_apply}(\mathbf{q}_{mug}, \mathbf{v}_{up\_world})$
  - $dot_{orient} = \mathbf{v}_{up\_mug} \cdot \mathbf{v}_{up\_world}$
- **獎勵公式**:
  $$ R_{orient} = \begin{cases} dot_{orient}^2 & \text{if is\_grasping} \land (h_{lift} > 0.01) \\ 0 & \text{otherwise} \end{cases} $$

---

#### 4.6. 動作懲罰 ($P_{action}$)

懲罰過大的動作指令。

- **懲罰公式**:
  $$ P_{action} = \sum_{i} a_i^2 $$
  其中 $a_i$ 是動作向量的第 $i$ 個分量。

## 5. 重製條件 (Reset Conditions)

當以下任一情況發生時，回合結束並重製環境：

1.  **任務成功**: 馬克杯被提起超過 `0.1` 米。
   - $h_{lift} > 0.1$
2.  **任務失敗 (掉落)**: 馬克杯的位置低於 `0.1` 米 (表示已從桌上掉落)。
   - $p_{mug, z} < 0.1$
3.  **時間耗盡**: 回合步數達到設定的最大值 `max_episode_length`。

## 6. 主要超參數 (Key Hyperparameters)

以下是在 `cfg` 文件中定義的、控制獎勵和環境行為的關鍵超參數：

- **獎勵權重**:
  - `dist_reward_scale` ($w_{dist}$)
  - `rot_reward_scale` ($w_{rot}$)
  - `lift_reward_scale` ($w_{lift}$)
  - `orientation_reward_scale` ($w_{orient}$)
  - `action_penalty_scale` ($w_{action}$)
- **環境參數**:
  - `episodeLength`
  - `actionScale`
  - `startPositionNoise`
  - `startRotationNoise`
  - `diabloDofNoise`
