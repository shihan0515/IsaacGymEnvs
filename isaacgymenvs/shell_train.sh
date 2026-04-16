#!/bin/bash
args=("42" "7" "123" "2021" "0" "193" "397")

GPU_ID=0  # Specify which GPU to use

for arg in "${args[@]}"
do
    for num_actor in {2..5}
    do
        exp_name="dumbbell_grasp_moe_num_actor_${num_actor}_seed_${arg}"
        echo "Starting: Num Actor: $num_actor Seed: $arg"

        CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
            task=DiabloGraspCustom3 \
            headless=True \
            experiment="$exp_name" \
            seed="$arg" \
            moe_num_actors="$num_actor"

        if [ $? -ne 0 ]; then
            echo "ERROR: Training failed for $exp_name"
        fi
    done
done
