import argparse
import itertools
import json
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import IO, Any, Callable, Iterator, List, Mapping

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import tqdm

from automation.api.audio import Audio
from automation.api.conversation import Conversation
from automation.api.out_trait import OutTrait
from automation.api.paragraph import Paragraph
from automation.factories.audio import AudioFactory
from automation.factories.pt import PtFactory
from automation.factories.sft import SftFactory
from automation.factories.vendor import VendorFactory
from automation.sdk.auto_format import (
    get_cyrene_whitelist_auto_format,
    get_retain_sentence_name_auto_format,
)
from automation.sdk.config import sentence_global_config
from automation.sdk.match_sub import (
    get_fast_felysneko_match_sub,
    get_felysneko_match_sub,
    get_slow_path_only_match_sub,
)
from automation.sdk.token_counter import get_qwen3_token_counter

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


def to_jsonl(
    iterable: Iterator[OutTrait], f: IO[str], **kwargs: Any
) -> Iterator[OutTrait]:
    for each in iterable:
        line = each.to_jsonl(**kwargs)
        json.dump(line, f, ensure_ascii=False, default=str)
        print(file=f)
        yield each


def num_tokens(iterable: Iterator[OutTrait]) -> Iterator[int]:
    for each in iterable:
        yield each.num_tokens


def emit(
    iterable: Iterator[OutTrait], output_path: Path, desc: str, **kwargs: Any
) -> List[int]:
    with open(output_path, "w+") as file:
        iterable = to_jsonl(iterable, file, **kwargs)
        iterable = num_tokens(iterable)
        metrics = list(tqdm.tqdm(iterable, desc=f"> {desc}"))
    return metrics


def __split_paragraphs(
    iterable: Iterator[Paragraph], max_token: int
) -> Iterator[Paragraph]:
    for each in iterable:
        yield from each.split(max_token)


def __clip_conversations(
    iterable: Iterator[Conversation], max_user_lines: int
) -> Iterator[Conversation]:
    for each in iterable:
        yield each.clip(max_user_lines)


def __split_conversations(
    iterable: Iterator[Conversation], max_token: int
) -> Iterator[Conversation]:
    for each in iterable:
        yield from each.split(max_token)


def __filter_conversations(
    iterable: Iterator[Conversation], max_token: int
) -> Iterator[Conversation]:
    for each in iterable:
        if each.num_tokens > max_token or each.is_self_talk():
            continue
        yield each


def __convert_audio(
    iterable: Iterator[Audio], output_dir: Path, num_threads: int
) -> Iterator[Audio]:
    def worker(audio: Audio) -> Audio:
        audio.wem_to_wav_by_vgmstream(output_dir)
        return audio

    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        for each in ex.map(worker, iterable):
            yield each


def pt_everything(
    output_dir: Path, turn_based_game_data_dir: Path, language: str
) -> List[int]:
    sentence_global_config(
        auto_format=get_retain_sentence_name_auto_format(language),
        token_counter=get_qwen3_token_counter(),
        match_sub=get_felysneko_match_sub(language),
    )

    target = PtFactory(turn_based_game_data_dir, language).everything.values()
    iterable = itertools.chain.from_iterable(target)
    iterable = __split_paragraphs(iterable, 4096)
    return emit(iterable, output_dir / f"{language}.jsonl", f"{language:>3}")


def pt_amphoreus(
    output_dir: Path, turn_based_game_data_dir: Path, language: str
) -> List[int]:
    sentence_global_config(
        auto_format=get_retain_sentence_name_auto_format(language),
        token_counter=get_qwen3_token_counter(),
        match_sub=get_felysneko_match_sub(language),
    )

    target = PtFactory(turn_based_game_data_dir, language).amphoreus.values()
    iterable = itertools.chain.from_iterable(target)
    iterable = __split_paragraphs(iterable, 4096)
    return emit(iterable, output_dir / f"{language}.jsonl", f"{language:>3}")


def pt_vendor(output_dir: Path, vendor_dir: Path):
    sentence_global_config(
        auto_format=None,
        token_counter=get_qwen3_token_counter(),
        match_sub=get_fast_felysneko_match_sub(),
    )

    factory = VendorFactory(vendor_dir)
    named_metrics = {
        "miyoushe": emit(
            factory.build_miyoushe(), output_dir / "miyoushe.jsonl", "miyoushe"
        ),
        "coig": emit(factory.build_coig(), output_dir / "coig.jsonl", "coig"),
    }

    return named_metrics


def sft_everything(
    output_dir: Path, turn_based_game_data_dir: Path, language: str
) -> List[int]:
    sentence_global_config(
        auto_format=None,
        token_counter=get_qwen3_token_counter(),
        match_sub=get_felysneko_match_sub(language),
    )

    target = SftFactory(turn_based_game_data_dir, language).everything.values()
    iterable = itertools.chain.from_iterable(target)
    iterable = __clip_conversations(iterable, 10)
    iterable = __split_conversations(iterable, 4096)
    iterable = __filter_conversations(iterable, 4096)
    return emit(
        iterable, output_dir / f"{language}.jsonl", f"{language:>3}", use_system=True
    )


def sft_amphoreus(
    output_dir: Path, turn_based_game_data_dir: Path, language: str
) -> List[int]:
    sentence_global_config(
        auto_format=None,
        token_counter=get_qwen3_token_counter(),
        match_sub=get_felysneko_match_sub(language),
    )

    target = SftFactory(turn_based_game_data_dir, language).amphoreus.values()
    iterable = itertools.chain.from_iterable(target)
    iterable = __clip_conversations(iterable, 10)
    iterable = __split_conversations(iterable, 4096)
    iterable = __filter_conversations(iterable, 4096)
    return emit(
        iterable, output_dir / f"{language}.jsonl", f"{language:>3}", use_system=True
    )


def sft_cyrene(
    output_dir: Path, turn_based_game_data_dir: Path, language: str
) -> List[int]:
    sentence_global_config(
        auto_format=get_cyrene_whitelist_auto_format(language),
        token_counter=get_qwen3_token_counter(),
        match_sub=get_felysneko_match_sub(language),
    )

    target = SftFactory(turn_based_game_data_dir, language).cyrene.values()
    iterable = itertools.chain.from_iterable(target)
    iterable = __clip_conversations(iterable, 10)
    iterable = __filter_conversations(iterable, 8192)
    return emit(
        iterable, output_dir / f"{language}.jsonl", f"{language:>3}", use_system=False
    )


def tts_cyrene(
    output_dir: Path,
    unpacked_audio_dir: Path,
    turn_based_game_data_dir: Path,
    language: str,
    num_threads: int,
) -> Iterator[Audio]:
    sentence_global_config(
        auto_format=None,
        token_counter=None,
        match_sub=get_slow_path_only_match_sub(),
    )

    unpacked_audio_language_dir = unpacked_audio_dir / language
    audio_dir = output_dir / language
    __ensure_dir_exist(unpacked_audio_language_dir)
    __ensure_dir_exist(audio_dir)

    target = AudioFactory(
        unpacked_audio_language_dir, turn_based_game_data_dir
    ).cyrene.values()
    iterable = itertools.chain.from_iterable(target)
    iterable = __convert_audio(iterable, audio_dir, num_threads)
    return emit(iterable, output_dir / f"{language}.jsonl", f"{language:>12}")


def __ensure_dir_exist(output_dir: Path) -> None:
    os.makedirs(output_dir, exist_ok=True)


def __dump_name_metrics(
    named_metrics: Mapping[str, List[int]], output_path: Path
) -> None:
    with open(output_path, "w+") as f:
        json.dump(named_metrics, f)

    num_tokens = sum(sum(x) for x in named_metrics.values())
    print(f"Estimated number of tokens: {num_tokens}")


def entry_multilingual(
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


def entry_vendor(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir / "pt" / "vendor"
    __ensure_dir_exist(output_dir)

    named_metrics = pt_vendor(output_dir, args.vendor_dir)

    __dump_name_metrics(named_metrics, output_dir.parent / "vendor.json")


def entry_audio(
    args: argparse.Namespace, f: Callable[[Path, Path], Iterator[Audio]]
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
                    entry_multilingual(args, pt_everything)
                case "pt", "amphoreus":
                    entry_multilingual(args, pt_amphoreus)
                case "sft", "everything":
                    entry_multilingual(args, sft_everything)
                case "sft", "amphoreus":
                    entry_multilingual(args, sft_amphoreus)
                case "sft", "cyrene":
                    entry_multilingual(args, sft_cyrene)
                case _:
                    parser.error(f"Unsupported: {args.namespace} {args.dataset}")
        case "vendor":
            entry_vendor(args)
        case "audio":
            match args.dataset:
                case "cyrene":
                    entry_audio(args, tts_cyrene)
                case _:
                    parser.error(f"Unsupported: {args.dataset}")


if __name__ == "__main__":
    main()
