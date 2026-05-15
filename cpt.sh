CUDA_VISIBLE_DEVICES=0 \
MODELSCOPE_CACHE=/root/autodl-tmp/.cache \
nohup swift pt \
    --torch_dtype 'bfloat16' \
    --model 'Qwen/Qwen3.5-4B-Base' \
    --model_type 'qwen3_5' \
    --template 'qwen3_5' \
    --dataset 'cpt.jsonl' 'cpt.vendor.jsonl' \
    --split_dataset_ratio '0.01' \
    --max_length '5120' \
    --tuner_type 'full' \
    --learning_rate '3e-5' \
    --num_train_epochs '3.0' \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 12 \
    --eval_steps '50' \
    --save_steps 200 \
    --output_dir '/root/autodl-tmp/cpt' \
    --attn_impl 'flash_attention_2' \
    --neftune_noise_alpha 0 \
    --truncation_strategy 'right' \
    --warmup_ratio '0.05' \
    --report_to 'tensorboard' \
    --use_liger_kernel 'True' \
    --dataset_num_proc 2 \
    --dataloader_num_workers 2 \
    --packing \
    --save_only_model \
    > "$(date +%s).log" 2>&1 &
