import argparse
import json
import multiprocessing as mp
import pandas as pd
import os
from pathlib import Path
from typing import Callable, List, Mapping

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
pd.set_option("future.no_silent_downcasting", True)

from automation.implementation.task import pt, sft, tts

TEXT_LANGUAGES = [
    "chs",
    "cht",
    "de",
    "en",
    "es",
    "fr",
    "id",
    "jp",
    "kr",
    "pt",
    "ru",
    "th",
    "vi",
]

VOICEOVER_LANGUAGES = [
    "Chinese(PRC)",
    "English",
    "Japanese",
    "Korean",
]


def __ensure_dir_exist(output_dir: Path) -> None:
    os.makedirs(output_dir, exist_ok=True)


def __dump_name_metrics(
    named_metrics: Mapping[str, List[int]], output_path: Path
) -> None:
    with open(output_path, "w+") as f:
        json.dump(named_metrics, f)

    num_tokens = sum(sum(x) for x in named_metrics.values())
    print(f"Estimated number of tokens: {num_tokens}")


def __multilingual(
    args: argparse.Namespace, f: Callable[[Path, Path, str], List[int]]
) -> None:
    output_dir: Path = args.output_dir / args.namespace / args.dataset
    __ensure_dir_exist(output_dir)

    tasks = [
        (output_dir, args.turn_based_game_data_dir, lang) for lang in TEXT_LANGUAGES
    ]
    pool = mp.Pool(processes=args.num_proc)

    try:
        metrics = pool.starmap(f, tasks)
    finally:
        pool.close()
        pool.join()

    named_metrics = {k: v for k, v in zip(TEXT_LANGUAGES, metrics)}
    __dump_name_metrics(named_metrics, output_dir.parent / f"{args.dataset}.json")


def __vendor(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir / "pt" / "vendor"
    __ensure_dir_exist(output_dir)

    named_metrics = pt.vendor(output_dir, args.vendor_dir)

    __dump_name_metrics(named_metrics, output_dir.parent / "vendor.json")


def __audio(
    args: argparse.Namespace, f: Callable[[Path, Path, Path, str, int], List[int]]
) -> None:
    output_dir: Path = args.output_dir / "tts" / args.dataset
    __ensure_dir_exist(output_dir)

    tasks = [
        (
            output_dir,
            args.unpacked_audio_dir,
            args.turn_based_game_data_dir,
            lang,
            args.num_threads,
        )
        for lang in VOICEOVER_LANGUAGES
    ]
    pool = mp.Pool(processes=len(VOICEOVER_LANGUAGES))

    try:
        metrics = pool.starmap(f, tasks)
    finally:
        pool.close()
        pool.join()

    named_metrics = {k: v for k, v in zip(VOICEOVER_LANGUAGES, metrics)}
    __dump_name_metrics(named_metrics, output_dir.parent / f"{args.dataset}.json")


def main() -> None:
    parser = argparse.ArgumentParser()

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--output-dir",
        type=Path,
        default=Path("corpora"),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    multilingual = subparsers.add_parser("multilingual", parents=[parent])
    multilingual.add_argument(
        "--turn-based-game-data-dir",
        type=Path,
        required=True,
    )
    multilingual.add_argument(
        "--namespace",
        choices=["pt", "sft"],
        required=True,
    )
    multilingual.add_argument(
        "--dataset",
        choices=["everything", "amphoreus", "cyrene"],
        required=True,
    )
    multilingual.add_argument(
        "--num-proc",
        type=int,
        default=4,
    )

    vendor = subparsers.add_parser("vendor", parents=[parent])
    vendor.add_argument(
        "--vendor-dir",
        type=Path,
        required=True,
    )

    audio = subparsers.add_parser("audio", parents=[parent])
    audio.add_argument(
        "--turn-based-game-data-dir",
        type=Path,
        required=True,
    )
    audio.add_argument(
        "--unpacked-audio-dir",
        type=Path,
        required=True,
    )
    audio.add_argument(
        "--dataset",
        choices=["cyrene"],
        required=True,
    )
    audio.add_argument(
        "--num-threads",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    match args.command:
        case "multilingual":
            match args.namespace, args.dataset:
                case "pt", "everything":
                    __multilingual(args, pt.everything)
                case "pt", "amphoreus":
                    __multilingual(args, pt.amphoreus)
                case "sft", "everything":
                    __multilingual(args, sft.everything)
                case "sft", "amphoreus":
                    __multilingual(args, sft.amphoreus)
                case "sft", "cyrene":
                    __multilingual(args, sft.cyrene)
                case _:
                    parser.error(f"Unsupported: {args.namespace} {args.dataset}")
        case "vendor":
            __vendor(args)
        case "audio":
            match args.dataset:
                case "cyrene":
                    __audio(args, tts.cyrene)
                case _:
                    parser.error(f"Unsupported: {args.dataset}")
        case _:
            parser.error(f"Unsupported: {args.command}")


if __name__ == "__main__":
    main()
