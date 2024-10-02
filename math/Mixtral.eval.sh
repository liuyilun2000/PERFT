#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=32:59:00
#SBATCH --mail-type=ALL
#SBATCH --job-name=OLMoE-1B-7B.lora.eval
#SBATCH --output=log/OLMoE-1B-7B.lora.%j.eval.slurm.log

# Check if a model name argument is provided
if [ $# -eq 0 ]; then
    echo "Please provide a model name argument"
    exit 1
fi

MODEL_NAME=$1

run_evaluation() {
    local dataset=$1
    python math_evaluate.py \
        --dataset $dataset \
        --base_model mistralai/Mixtral-8x7B-v0.1 \
        --peft_model checkpoints/Mixtral-8x7B-v0.1.${MODEL_NAME} \
        --name Mixtral-8x7B.${MODEL_NAME} \
        --batch_size 8 --max_new_tokens 256 \
        | tee -a log/Mixtral-8x7B.${MODEL_NAME}.eval.${dataset}.log
}

for dataset in "MultiArith" "gsm8k" "AddSub" "AQuA" "SingleEq" "SVAMP"; do
    run_evaluation $dataset
done