CUDA_VISIBLE_DEVICES=0 \
MODELSCOPE_CACHE=/root/autodl-tmp/.cache \
swift infer \
    --adapters /root/autodl-tmp/sft/v2-20260512-054327/checkpoint-500 \
    --system '和银河猫猫侠在一起。' \
    --stream true \
    --temperature 0.5 \
    --top_p 0.9 \
    --enable_thinking false \
    --max_new_tokens 1024
