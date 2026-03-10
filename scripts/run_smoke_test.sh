#!/bin/bash
set -e
cd /home/yzy/Documents/GitHub/prac
source .venv/bin/activate

echo "[INFO] Python: $(python --version)"
echo "[INFO] Starting PRAC fine-tuning smoke test..."

python scripts/finetune_llama.py \
  --model_name /data/models/Llama-3.1-8B-Instruct/ \
  --dataset alpaca \
  --output_dir ./output \
  --use_prac \
  --principal_rank 0.25 \
  --random_rank 0.25 \
  --prac_update_freq 200 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --num_epochs 1 \
  --max_seq_length 512 \
  --max_train_samples 500 \
  --bf16 \
  --gradient_checkpointing \
  --seed 42

echo "[INFO] Fine-tuning complete!"
