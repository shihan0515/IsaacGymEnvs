# Diablo Grasping Task (Custom v3)

本專案實作了一個基於 Isaac Gym 的強化學習環境，控制 **Diablo 二輪足式機器人** 執行複雜的「抓取、搬運、放置」任務。

## 任務目標
1. **靠近並對準**：控制右手臂靠近並對齊桌面上的 YCB 電鑽 (Power Drill)。
2. **抓取與抬升**：精準抓取電鑽手把並將其抬起。
3. **平台對準**：電鑽被抬起後，會出現一個**橘色漂浮平台**，機器人須將電鑽搬運至平台上方。
4. **放置與撤離**：將電鑽垂直放置於平台上，打開夾爪，並將手臂完全撤離以完成任務。

## 環境需求 (Prerequisites)
* **Isaac Gym**: 已安裝 NVIDIA Isaac Gym (建議版本 Preview 4)。
* **IsaacGymEnvs**: 本任務建構於 `IsaacGymEnvs` 架構下。
* **Python**: 3.7+ (建議使用 Conda 環境)。
* **Git LFS**: 本專案使用 Git LFS 儲存大型 URDF 與 Mesh 檔案。

## 資產準備 (Assets Setup)
確保以下資產放置於正確的路徑：
1. **Diablo 機器人**: 
   `assets/urdf/diab/Diablo_ger/Part_gripper_col_rev/URDF/diablo_erc1.urdf`
2. **YCB 電鑽**: 
   `assets/urdf/ycb/ycb_urdfs-main/ycb_assets/035_power_drill.urdf`

> **注意**：請確保已執行 `git lfs pull` 以獲取完整的 Mesh 檔案。

## 如何執行 (How to Run)

### 1. 註冊任務
確保在 `isaacgymenvs/tasks/__init__.py` 中已加入以下註冊代碼：
```python
from .diablo_graspcustom3 import DiabloGraspCustom3
```
並在 `isaacgym_task_map` 字典中加入：
```python
"DiabloGraspCustom3": DiabloGraspCustom3,
```

### 2. 啟動訓練 (RL Training)
使用預設的 PPO 演算法進行訓練：
```bash
python train.py task=DiabloGraspCustom3
```

### 3. 查看模擬 (僅測試環境)
如果您只想查看機器人初始化與環境配置，可以執行：
```bash
python train.py task=DiabloGraspCustom3 headless=False num_envs=16
```

### 4. 載入模型進行測試 (Play/Inference)
如果您已有訓練好的模型（例如位於 `runs/` 下）：
```bash
python train.py task=DiabloGraspCustom3 checkpoint=runs/path_to_model/model.pth test=True num_envs=1
```

## 任務特點 (Key Features)
* **非對稱隨機化**：機器人底盤高度與桌面高度會同步但非對稱地隨機化（機器人升 1cm，桌子升 0.5cm），測試 Agent 的泛化能力。
* **動態平台**：目標放置平台僅在物體被成功舉起（高度 > 2cm）後才會動態出現。
* **階段性獎勵機制**：包含靠近、抓取、垂直姿態、放置、釋放與撤離等多階段獎勵。
* **高度與旋轉隨機化**：針對底盤高度、俯仰角（Pitch）及桌面旋轉進行了範圍優化，以平衡學習難度。
