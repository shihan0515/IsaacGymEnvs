# 實驗流程筆記

工作目錄：`/home/erc/isaacgym/python/IsaacGymEnvs/isaacgymenvs/`

---

## 整體流程概覽

```
1. 訓練  →  2. 評估  →  3. 解析 log  →  4. 畫圖
```

---

## 第一步：訓練

### 腳本：`shell_train.sh`

用多個 seed 依序訓練，每個 seed 產生一個獨立的 checkpoint。

**關鍵設定：**
```bash
args=("42" "7" "123" "2021" "0" "193" "397")  # 訓練 seed 列表
for num_actor in 2                              # MoE actor 數量（1=PPO, 2=MoE-PPO）
```

**執行：**
```bash
bash shell_train.sh
```

**輸出：** `runs/<實驗名稱>_<時間戳>/nn/<實驗名稱>.pth`

**注意事項：**
- 7 個 seed × 1 個 num_actor = 共 7 次訓練，依序執行
- 若要同時訓練 PPO（num_actor=1）和 MoE-PPO（num_actor=2），改為 `for num_actor in 1 2`，共 14 次
- checkpoint 路徑在訓練結束後的 log 中可找到

---

## 第二步：評估

### 情境 A：單一模型、整體評估
**腳本：`shell_evaluate.sh`**

對一個 checkpoint 用多 seed 做整體評估（混合三種物件）。

**關鍵設定：**
```bash
EXP_NAME="runs/.../nn/xxx.pth"   # 要評估的 checkpoint 路徑
NUM_ACTORS=2                       # 對應訓練時的 moe_num_actors
```

**執行：**
```bash
bash shell_evaluate.sh
```

**輸出：** `evaluation_logs_<時間戳>/seed_<N>.log` + `summary_results.txt`

---

### 情境 B：泛化模型、分物件評估（PPO vs MoE-PPO 同時跑）
**腳本：`shell_evaluate_generalization.sh`**

同時評估 PPO 和 MoE-PPO 兩個 checkpoint，對 mug / drill / dumbbell 分開測試。

**關鍵設定：**
```bash
SEEDS=("42" "7" "123" "2021" "0")
PPO_CKPT="runs/.../nn/xxx.pth"
MOE_CKPT="runs/.../nn/xxx.pth"
PPO_ACTORS=1
MOE_ACTORS=2
```

**執行：**
```bash
bash shell_evaluate_generalization.sh
```

**輸出：** `eval_generalization_<時間戳>/`
- `{object}_{model}_seed{N}.log`（每次評估的完整 log）
- `summary.txt`（彙整所有結果）

**總執行次數：** 3 物件 × 2 模型 × 5 seeds = 30 次

---

### 情境 C：指定特定物件評估
**腳本：`shell_evaluate_specific.sh`**

對單一物件（或全部混合）做多 seed 評估。

**執行：**
```bash
bash shell_evaluate_specific.sh mug       # 只測 mug
bash shell_evaluate_specific.sh drill     # 只測 drill
bash shell_evaluate_specific.sh dumbbell  # 只測 dumbbell
bash shell_evaluate_specific.sh all       # 混合三種（預設）
```

---

## 第三步：解析 log → Excel

### 情境 A：單一模型 log（`shell_evaluate.sh` 產出）
**腳本：`parse_evaluation_results.py`**

```bash
python parse_evaluation_results.py evaluation_logs_<時間戳>/
```

**輸出：** 終端印出可複製貼入 Excel 的 TSV 表格 + `aggregate_results.csv`

---

### 情境 B：泛化評估 log（`shell_evaluate_generalization.sh` 產出）
**腳本：`parse_generalization_to_excel.py`**

```bash
python parse_generalization_to_excel.py eval_generalization_<時間戳>/ /home/erc/Downloads/pp_evaluation_generalization.xlsx
```

**輸出：** Excel 檔案，三個 sheet（mug / drill / dumbbell），格式：

| 欄位 | 說明 |
|---|---|
| seed | 評估 seed |
| ppo_sr / moe_sr | 成功率 (%) |
| ppo_l2 / moe_l2 | 放置距離 (mm) |
| ppo_avg_reward / moe_avg_reward | 平均 reward |
| ppo_avg_steps / moe_avg_steps | 平均步數 |
| Average | 各欄平均值 |
| improve | MoE-PPO 相對 PPO 的改善率 |

---

## 第四步：畫圖

### 訓練曲線
**腳本：`plot_reward_curves_local.py`**

從 TensorBoard event 檔讀取訓練過程數據，畫訓練曲線。

**執行：**
```bash
MPLBACKEND=Agg python plot_reward_curves_local.py
```

**輸出：** `figures/`
- `mix_total_reward.png` — 泛化訓練總 reward 曲線
- `mix_success_rate.png` — 泛化訓練成功率曲線
- `mix_placement_dist.png` — 泛化訓練放置距離曲線
- `drill_total_rewards.png` / `mug_total_rewards.png` / `dumbbell_total_rewards.png` — 單物件訓練曲線

**關鍵設定（腳本頂部）：**
```python
mix_checkpoints = [
    "runs/<PPO runs>/summaries/events.out.tfevents.*",
    "runs/<MoE-PPO runs>/summaries/events.out.tfevents.*",
]
```

---

### 評估結果長條圖
**腳本：`evaluation_pp_local.py`**

從 Excel 讀取評估統計，畫 PPO vs MoE-PPO 比較長條圖。

**執行：**
```bash
MPLBACKEND=Agg python evaluation_pp_local.py
```

**輸出：** `figures/`
- `plot_success_rate.png` — 單物件成功率比較
- `plot_l2_dist.png` — 單物件放置距離比較
- `plot_gen_success_rate.png` — 泛化成功率比較
- `plot_gen_l2_dist.png` — 泛化放置距離比較

**關鍵設定（腳本頂部）：**
```python
XLSX_PATH = "/home/erc/Downloads/pp_evaluation.xlsx"           # 單物件結果
XLSX_GEN  = "/home/erc/Downloads/pp_evaluation_generalization.xlsx"  # 泛化結果
```

---

## 實驗設計說明

### 為什麼需要多個訓練 seed？
同一演算法用不同 seed 訓練，結果會有些微差異（初始權重不同）。
多個訓練 seed 取平均，才能排除「運氣好」的情況，確保比較有統計意義。

### 訓練 seed vs 評估 seed 的差別
| | 訓練 seed | 評估 seed |
|---|---|---|
| 目的 | 排除初始化運氣 | 排除環境隨機性 |
| 影響 | checkpoint 品質 | 測試時的環境分布 |

### 正確比較 PPO vs MoE-PPO 的方式
```
PPO  訓練 seed [42,7,123,...] → 7 個 checkpoint → 各自多 seed 評估 → 平均成功率 ± std
MoE-PPO 同上
最終：比較兩組平均值，可做 t-test 檢定顯著性
```

---

## 現有實驗資料位置

| 資料 | 路徑 |
|---|---|
| 單物件評估結果 | `/home/erc/Downloads/pp_evaluation.xlsx` |
| 泛化評估結果 | `/home/erc/Downloads/pp_evaluation_generalization.xlsx` |
| 泛化評估 log | `eval_generalization_20260501_111214/` |
| 輸出圖片 | `figures/` |
| PPO 泛化 checkpoint | `runs/DiabloGraspCustom3_29-22-48-04/nn/DiabloGraspCustom3.pth` |
| MoE-PPO 泛化 checkpoint | `runs/DiabloGraspCustom3_30-14-28-47/nn/DiabloGraspCustom3.pth` |
