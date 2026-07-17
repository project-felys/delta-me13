# δ-me13

δ-me13 generates text and audio datasets for Cyrene, and can be extended to a wider range of applications. Feel free to [chat](https://felys.dev/en/chat) with Cyrene and read the [blog](https://book.felys.dev/bring-cyrene-to-life.html). This repository contains the training details, scripts, and Dockerfile for deployment. The trained weights are available here: [Delta-me13](https://huggingface.co/FelysNeko/Delta-me13), [PhiLia093](https://huggingface.co/FelysNeko/PhiLia093).

**Note: this project does not contain any game assets, nor does it reverse engineer any encrypted files.**

## Large-Language-Model

The corpus generation scripts depend on an external game data repository, which I will not name here. If you find that repository, clone it and replace `path/to/game-data-repository` with its path. The following commands build the dataset in [standard](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html#standard-dataset-format) format for all 13 languages.

```bash
REPO=path/to/game-data-repository

# Pre-training
python3 main.py multilingual \
    --turn-based-game-data-dir $REPO \
    --namespace pt \
    --dataset amphoreus \
    --num-proc 13

# Supervised Fine-tuning
python3 main.py multilingual \
    --turn-based-game-data-dir $REPO \
    --namespace sft \
    --dataset cyrene
```

The vendor data includes the official game [wiki](https://bbs.mihoyo.com/sr/wiki/content/5851/detail) and LeetCode problems from [COIG](https://huggingface.co/datasets/BAAI/COIG/blob/main/leetcode_instructions.jsonl), so no external data source is required.

```bash
# Pre-training
python3 main.py vendor --vendor-dir vendor
```

## Text-to-Speech

Generating the audio dataset requires both the external repository (see the previous [section](#large-language-model)) and the game itself. You will need to unpack the audio files first. Replace `path/to/persistent/audio/audio-package/windows` with the game's audio directory.

```bash
PCK=path/to/persistent/audio/audio-package/windows

# Unpack *.pck
python3 pck.py --input-dir $PCK
```

Once all `.pck` files have been unpacked, the command-line interface can process the corpus and generate `.wav` files. Make sure `vgmstream-cli` is in your `PATH`, or run `export VGMSTREAM=path/to/vgmstream-cli`. Refer to [vgmstream](https://github.com/vgmstream/vgmstream) for installation guidance.

```bash
REPO=path/to/game-data-repository

# Text-to-Speech
python3 main.py audio \
    --turn-based-game-data-dir $REPO \
    --unpacked-audio-dir audio \
    --dataset cyrene
```

## License

Distributed under the terms of the [LICENSE](LICENSE).

## Copyright

© All rights reserved by FelysNeko.
