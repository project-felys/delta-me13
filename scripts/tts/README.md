# Examples

## Preprocess

```bash
python3 into_swift_format.py \
  --jsonl "cyrene/Chinese(PRC).jsonl" \
  --output-dir audio/raw
```

```bash
python3 precompute_audio_codes.py \
  --device cuda:0 \
  --tokenizer-model-path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input-jsonl "audio/raw/Chinese(PRC).jsonl" \
  --output-dir audio
```

```bash
python3 split_train_test.py \
  --jsonl "audio/Chinese(PRC).jsonl"
```

## Train

```shell
nohup python3 vendor/bo_loop.py \
    --tuner-type lora \
    --train-jsonl "audio/train/0/Chinese(PRC).jsonl" \
    --test-jsonl "audio/test/0/Chinese(PRC).jsonl" \
    --speaker-name cyrene \
    --output-model-path "output/lora" \
    > "$(date +%s).log" 2>&1 &
```

```shell
nohup python3 vendor/trainer.py \
    --tuner-type full \
    --train-jsonl "audio/Chinese(PRC).jsonl" \
    --speaker-name cyrene \
    --init-model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --output-model-path "output/chosen/full/outcome" \
    --batch-size 4 \
    --gradient-accumulation-steps 1 \
    --num-epochs 5 \
    --lr 6.59e-6 \
    --min-lr-factor 0.404 \
    --warmup-ratio 0.05 \
    > "$(date +%s).log" 2>&1 &
```

```shell
nohup python3 vendor/trainer.py \
    --tuner-type lora \
    --train-jsonl "audio/Chinese(PRC).jsonl" \
    --speaker-name cyrene \
    --init-model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --output-model-path "output/chosen/lora/outcome" \
    --batch-size 16 \
    --gradient-accumulation-steps 1 \
    --num-epochs 4 \
    --lr 4.29e-4 \
    --min-lr-factor 0.490 \
    --lora-rank 8 \
    --lora-alpha 8 \
    --warmup-ratio 0.05 \
    > "$(date +%s).log" 2>&1 &
```

## Evaluation

```bash
python3 vendor/gen_audio.py \
    --test-jsonl "audio/test/0/Chinese(PRC).jsonl" \
    all \
    --model "output/chosen/selected" \
    --speaker-name cyrene
```

```bash
python3 vendor/gen_audio.py \
    --test-jsonl "audio/test/0/Chinese(PRC).jsonl" \
    baseline \
    --output-path "output/chosen/baseline/0/test" \
    --ref-audio "audio/raw/ref.wav" \
    --ref-text "三千万篇相似而不相同的故事，听起来是不是很多呢？可我从来没有厌倦过，和你旅行的时候，又感动地重温了一遍呢！但故事的每一页，似乎都有些遗憾，想在这里加一段欢笑，想在那里填补上空白…多亏有你在，终于能为它们续写浪漫的结局了呀♪" \
    --language Auto
```

```bash
nohup python3 vendor/eval_nn.py \
    --ground-truth-jsonl "audio/test/0/Chinese(PRC).jsonl" \
    --test-dir \
        "output/chosen/baseline/0/test" \
        "output/chosen/full/0/test" \
        "output/chosen/lora/0/test" \
    > "$(date +%s).log" 2>&1 &
```
