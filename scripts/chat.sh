CUDA_VISIBLE_DEVICES=0 \
MODELSCOPE_CACHE=/root/autodl-tmp/.cache \
swift infer \
    --adapters /root/autodl-tmp/sft/v8-20260609-035234/checkpoint-363 \
    --system "你是昔涟，在陪银河猫猫侠聊天。" \
    --stream true \
    --temperature 0.5 \
    --top_p 0.8 \
    --top_k 10 \
    --enable_thinking false \
    --max_new_tokens 1024
