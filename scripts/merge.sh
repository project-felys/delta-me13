MODELSCOPE_CACHE=/root/autodl-tmp/.cache \
swift export \
    --adapters  /root/autodl-tmp/sft/v8-20260609-035234/checkpoint-363 \
    --merge_lora true
