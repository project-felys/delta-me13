# δ-me13

The first product of this project is `PhiLia093-4B-FP8-BLOCK`. Feel free to [chat](https://felys.dev/en/chat) with Cyrene. This repository contains training details, scripts, and the Dockerfile for deployment.

## Corpora

The corpora generation scripts rely on an external game data repository, which I will not name here. If you find that repository, clone it and replace `<game-data-repository>` with its path. The following commands build the dataset in [standard](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html#standard-dataset-format) format for all 13 languages.

```sh
# Pre-training
python3 main.py --dataset everything --root-dir <game-data-repository> --num-proc 13
python3 main.py --dataset amphoreus --root-dir <game-data-repository>

# Supervised Fine-turing
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
