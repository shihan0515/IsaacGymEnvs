#!/bin/bash
# 評價特定物品的腳本 (修正版)
# 用法: ./shell_evaluate_specific.sh [mug|drill|dumbbell|all]

TARGET_OBJ=${1:-"all"} # 如果沒傳參數，預設評測全部(混合)
args=("2021" "7" "123" "2021" "0" "193" "397") # 可以改為多個 seed，如 ("42" "123" "2021")

GPU_ID=0  # 指定 GPU
NUM_ENVS=1000

# === 根據你的需求設定模型路徑與 MoE 參數 ===
EXP_NAME="./runs/DiabloGraspCustom3_30-14-28-47/nn/DiabloGraspCustom3.pth"
MOE_ACTORS=2
# =========================================

# 建立日誌目錄
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="evaluation_logs_${TARGET_OBJ}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

SUMMARY_FILE="$LOG_DIR/summary_results.txt"
echo "Evaluation Summary for Object: $TARGET_OBJ" > "$SUMMARY_FILE"
echo "Model: $EXP_NAME" >> "$SUMMARY_FILE"
echo "MoE Actors: $MOE_ACTORS" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"

for arg in "${args[@]}"
do
    echo "------------------------------------------------"
    echo "Starting Evaluation: Obj=$TARGET_OBJ, Seed=$arg"
    LOG_FILE="$LOG_DIR/seed_${arg}.log"
    
    # 關鍵：透過 task.env.eval_object_name 傳入特定物品名稱
    EVAL_PARAM=""
    if [ "$TARGET_OBJ" != "all" ]; then
        EVAL_PARAM="task.env.eval_object_name=$TARGET_OBJ"
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
        task=DiabloGraspCustom3 \
        headless=True \
        num_envs=$NUM_ENVS \
        seed="$arg" \
        test=True \
        checkpoint="$EXP_NAME" \
        moe_num_actors="$MOE_ACTORS" \
        $EVAL_PARAM \
        2>&1 | tee "$LOG_FILE"
    
    # === 修正後的抓取邏輯 (對應 Python 實際輸出格式) ===
    echo "Seed: $arg" >> "$SUMMARY_FILE"
    # 抓取最後一次出現的指標整行內容
    grep "final success_rate:" "$LOG_FILE" | tail -n 1 >> "$SUMMARY_FILE" 2>/dev/null
    grep "final mean_placement_dist:" "$LOG_FILE" | tail -n 1 >> "$SUMMARY_FILE" 2>/dev/null
    grep "av reward:" "$LOG_FILE" | tail -n 1 >> "$SUMMARY_FILE" 2>/dev/null
    echo "" >> "$SUMMARY_FILE"
done

echo ""
echo "=========================================="
echo "Evaluation Results Summary:"
cat "$SUMMARY_FILE"
echo "=========================================="
echo "Done. Results saved in $LOG_DIR"
