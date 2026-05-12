CUDA_VISIBLE_DEVICES=0 \
MODELSCOPE_CACHE=/root/autodl-tmp/.cache \
nohup swift sft \
    --torch_dtype 'bfloat16' \
    --model '/root/autodl-tmp/cpt/v1-20260512-015051/checkpoint-400' \
    --model_type 'qwen3_5' \
    --template 'qwen3_5' \
    --dataset 'sft.jsonl' 'sft-extra.jsonl' \
    --split_dataset_ratio '0.01' \
    --max_length '4096' \
    --lora_rank '32' \
    --lora_alpha '64' \
    --learning_rate '1e-4' \
    --num_train_epochs '2.0' \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_steps '50' \
    --save_steps '100' \
    --output_dir '/root/autodl-tmp/sft' \
    --attn_impl 'flash_attention_2' \
    --neftune_noise_alpha '5' \
    --truncation_strategy 'right' \
    --warmup_ratio '0.05' \
    --report_to 'tensorboard' \
    --use_liger_kernel 'True' \
    --dataset_num_proc 2 \
    --dataloader_num_workers 2 \
    --packing \
    > "$(date +%s).log" 2>&1 &
