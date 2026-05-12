CUDA_VISIBLE_DEVICES=0 \
MODELSCOPE_CACHE=/root/autodl-tmp/.cache \
swift infer \
    --adapters /root/autodl-tmp/sft/v2-20260512-054327/checkpoint-500 \
    --system '铁幕之战后，终于和银河猫猫侠重逢了，一同回到哀丽秘榭，坐在秋千上闲聊。' \
    --stream true \
    --temperature 0.8 \
    --top_p 0.9 \
    --repetition_penalty 1.1 \
    --enable_thinking false \
    --max_new_tokens 1024
