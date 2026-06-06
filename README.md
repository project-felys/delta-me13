# δ-me13

The first product of this project is `PhiLia093-4B-FP8-BLOCK`. Feel free to [chat](https://felys.dev/en/chat) with Cyrene. This repository contains training details, scripts, and the Dockerfile for deployment.

## Corpora

The corpora generation scripts rely on an external game data repository, which I will not mention here. The command below builds the dataset in all languages:

```
python3 main.py --root-dir <game-data-repository> --mode <mode>
```

If you find the game data repository, clone it and replace `<game-data-repository>` with its path. Use `pt` or `sft` for the `--mode` argument: `pt` generates Amphoreus corpora for pre-training, while `sft` generates Cyrene corpora for supervised fine-tuning. Both outputs use ChatML format.

## License

Distributed under the terms of the [LICENSE](LICENSE).

## Copyright

© All rights reserved by FelysNeko
