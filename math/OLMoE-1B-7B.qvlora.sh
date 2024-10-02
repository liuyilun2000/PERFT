#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=15:59:00
#SBATCH --job-name=OLMoE-1B-7B.qvlora.train
#SBATCH --output=log/OLMoE-1B-7B.qvlora.%j.train.slurm.log
#SBATCH --mail-type=ALL
'''
declare -a configs=(
    "4 8 4"
    "8 16 8"
    "16 32 16"
    "32 64 32"
)
'''
declare -a configs=(
    "1 2 1"
    "2 4 2"
    "64 128 64"
    "128 256 128"
)
# Function to run training
run_training() {
    local lora_r=$1
    local lora_alpha=$2
    local run_name=$3
    local gpu_id=$4
    
    # Set environment variable for this GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    
    local full_run_name="OLMoE-1B-7B.qvlora_${run_name}"
    local output_dir="checkpoints/OLMoE-1B-7B-0924.qvlora_${run_name}"
    local log_file="log/${full_run_name}.train.log"
    
    echo "Starting training for ${full_run_name} on GPU ${gpu_id}"
    
    python finetune_qvlora.py \
        --base_model 'allenai/OLMoE-1B-7B-0924' \
        --data_path 'math_50k.json' \
        --output_dir "${output_dir}" \
        --batch_size 16 --micro_batch_size 16 --num_epochs 3 \
        --learning_rate 1e-5 --cutoff_len 256 --val_set_size 120 \
        --eval_step 80 --save_step 80 \
        --adapter_type 'LoRA' --lora_r ${lora_r} --lora_alpha ${lora_alpha} \
        --wandb_project 'peft-moe' \
        --wandb_run_name "${full_run_name}" \
        2>&1 | tee -a "${log_file}"
    
    echo "Finished training for ${full_run_name} on GPU ${gpu_id}"
}


# Run training tasks in parallel
for i in "${!configs[@]}"; do
    IFS=' ' read -ra config <<< "${configs[$i]}"
    run_training "${config[@]}" $i &
done

# Wait for all background jobs to finish
wait

echo "All training jobs completed."