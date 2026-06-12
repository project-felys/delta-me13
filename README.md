# δ-me13

δ-me13 is a project for large-language-model fine-tuning focused on role-playing, and includes a pipeline for constructing corpora. Feel free to [chat](https://felys.dev/en/chat) with Cyrene and read the [blog](https://book.felys.dev/bring-cyrene-to-life.html). This repository contains training details, scripts, and the Dockerfile for deployment. Here are the trained weights: [Delta-me13](https://huggingface.co/FelysNeko/Delta-me13), [PhiLia093](https://huggingface.co/FelysNeko/PhiLia093).

## Corpora

The corpora generation scripts rely on an external game data repository, which I will not name here. If you find that repository, clone it and replace `<game-data-repository>` with its path. The following commands build the dataset in [standard](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html#standard-dataset-format) format for all 13 languages.

```sh
# Pre-training
python3 main.py --dataset everything --root-dir <game-data-repository> --num-proc 13
python3 main.py --dataset amphoreus --root-dir <game-data-repository>

# Supervised Fine-tuning
python3 main.py --dataset cyrene --root-dir <game-data-repository>
```

Vendor data is copied from the official [wiki](https://bbs.mihoyo.com/sr/wiki/content/5851/detail), so no external data source is needed.

```sh
# Pre-training
python3 main.py --dataset vendor --root-dir vendor
```

## License

Distributed under the terms of the [LICENSE](LICENSE).

## Copyright

© All rights reserved by FelysNeko
