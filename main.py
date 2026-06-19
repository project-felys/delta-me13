import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


import argparse
import itertools
import json
import multiprocessing as mp
from pathlib import Path
from typing import IO, Any, Iterator, List, Literal, Mapping

from automation.api.conversation import Conversation
from automation.api.paragraph import Paragraph
from automation.api.sentence import Sentence
from automation.api.out_trait import OutTrait
from automation.factories.sft import SftFactory
from automation.factories.pt import PtFactory
from automation.factories.vendor import VendorFactory
from automation.match_sub import MatchSub

import tqdm
from transformers import AutoTokenizer

LANGUAGES = [
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


def get_token_counter():
    tokenizer = AutoTokenizer.from_pretrained(
        "tokenizer", trust_remote_code=True, local_files_only=True
    )
    return lambda x: len(tokenizer.encode(x))


def get_match_sub(language: str, is_male: bool, mute_warning: bool):
    nickname = "银河猫猫侠" if language.lower() in ("chs", "cht") else "FelysNeko"
    return MatchSub(nickname, is_male, mute_warning)


def get_auto_format(language: str, mode: Literal["pt", "sft", "vendor", "cyrene"]):
    lparen = "("
    rparen = ")"
    colon = ": "
    if language.lower() in ("chs", "cht"):
        lparen = "（"
        rparen = "）"
        colon = "："

    def pt_auto_format(sentence: Sentence) -> str:
        text = sentence.text
        if sentence.name is None:
            return text
        else:
            return f"{sentence.name}{colon}{text}"

    def sft_auto_format(sentence: Sentence) -> str:
        return sentence.text

    def cyrene_auto_format(sentence: Sentence) -> str:
        text = sentence.text.lstrip(lparen).rstrip(rparen)
        name_hash = sentence.name_hash
        if (
            name_hash is None
            or name_hash == 7108701208745180573
            or name_hash
            in {
                2309313067306506373,  # 昔涟
                11001695012879251917,  # 迷迷
                6462539533232001276,  # 少女的声音
                402745232924048698,  # 「记忆的花」
                14378608167795068022,  # 「记忆的花蕾」
                122935082492335341,  # 「记忆的幼芽」
                11263148736238834705,  # 「记忆的种子」
                6846009442988187041,  # {NICKNAME}&昔涟
                4455374036049186635,  # 昔涟
                13483973725572441939,  # 「另一位作者♪」
                12560857117710338840,  # 往昔的涟漪
            }
        ):
            return text
        else:
            return f"{lparen}{text}{rparen}"

    if mode == "pt" or mode == "vendor":
        return pt_auto_format
    elif mode == "sft":
        return sft_auto_format
    elif mode == "cyrene":
        return cyrene_auto_format
    else:
        raise


def setup_global_config(language: str, mode: Literal["pt", "sft", "vendor", "cyrene"]):
    auto_format = get_auto_format(language, mode)
    token_counter = get_token_counter()
    match_sub = get_match_sub(language, False, mode == "vendor")

    Sentence.set_auto_format_func(auto_format)
    Sentence.set_token_counter_func(token_counter)
    Sentence.set_match_sub_func(match_sub)


def to_jsonl(
    iterable: Iterator[OutTrait], use_system: bool, f: IO[str]
) -> Iterator[Paragraph]:
    for each in iterable:
        line = each.to_jsonl(use_system)
        json.dump(line, f, ensure_ascii=False)
        print(file=f)
        yield each


def num_tokens(iterable: Iterator[OutTrait]) -> Iterator[int]:
    for each in iterable:
        yield each.num_tokens


def emit(iterable: Iterator[OutTrait], output_path: Path, use_system: bool, desc: str):
    with open(output_path, "w+") as file:
        iterable = to_jsonl(iterable, use_system, file)
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
        if each.num_tokens <= max_token:
            yield each


def pt_honkai_star_rail(output_dir: Path, root_dir: Path, language: str):
    setup_global_config(language, "pt")
    target = PtFactory(root_dir, language).everything.values()
    iterable = itertools.chain.from_iterable(target)
    iterable = __split_paragraphs(iterable, 4096)
    return emit(iterable, output_dir / f"{language}.jsonl", False, f"{language:>3}")


def pt_amphoreus(output_dir: Path, root_dir: Path, language: str):
    setup_global_config(language, "pt")
    target = PtFactory(root_dir, language).amphoreus.values()
    iterable = itertools.chain.from_iterable(target)
    iterable = __split_paragraphs(iterable, 4096)
    return emit(iterable, output_dir / f"{language}.jsonl", False, f"{language:>3}")


def pt_vendor(output_dir: Path, root_dir: Path):
    setup_global_config("chs", "vendor")
    factory = VendorFactory(root_dir)
    return {
        "miyoushe": emit(
            factory.build_miyoushe(), output_dir / "miyoushe.jsonl", False, "miyoushe"
        ),
        "coig": emit(factory.build_coig(), output_dir / "coig.jsonl", False, "coig"),
    }


def sft_honkai_star_rail(output_dir: Path, root_dir: Path, language: str):
    setup_global_config(language, "sft")
    target = SftFactory(root_dir, language).everything.values()
    iterable = itertools.chain.from_iterable(target)
    iterable = __clip_conversations(iterable, 10)
    iterable = __split_conversations(iterable, 4096)
    iterable = __filter_conversations(iterable, 4096)
    return emit(iterable, output_dir / f"{language}.jsonl", True, f"{language:>3}")


def sft_cyrene(output_dir: Path, root_dir: Path, language: str):
    setup_global_config(language, "cyrene")
    target = SftFactory(root_dir, language).cyrene.values()
    iterable = itertools.chain.from_iterable(target)
    iterable = __clip_conversations(iterable, 10)
    return emit(iterable, output_dir / f"{language}.jsonl", False, f"{language:>3}")


def __ensure_output_dir(output_dir: Path):
    os.makedirs(output_dir, exist_ok=True)


def __dump_name_metrics(named_metrics: Mapping[str, List[int]], output_path: Path):
    with open(output_path, "w+") as f:
        json.dump(named_metrics, f)

    num_tokens = sum(sum(x) for x in named_metrics.values())
    print(f"Estimated number of tokens: {num_tokens}")


def entry_multilingual(args: argparse.Namespace, f: Any):
    output_dir = args.output_dir / args.dataset
    __ensure_output_dir(output_dir)

    tasks = [(output_dir, args.root_dir, lang) for lang in LANGUAGES]
    pool = mp.Pool(processes=args.num_proc)

    try:
        metrics = pool.starmap(f, tasks)
    finally:
        pool.close()
        pool.join()

    named_metrics = {k: v for k, v in zip(LANGUAGES, metrics)}
    __dump_name_metrics(named_metrics, args.output_dir / f"{args.dataset}.json")


def entry_vendor(args: argparse.Namespace):
    output_dir = args.output_dir / "vendor"
    __ensure_output_dir(output_dir)

    named_metrics = pt_vendor(output_dir, args.root_dir)

    __dump_name_metrics(named_metrics, args.output_dir / "vendor.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("corpora"),
    )
    parser.add_argument(
        "--dataset",
        choices=["pt", "amphoreus", "vendor", "sft", "cyrene"],
        required=True,
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    if args.dataset == "pt":
        entry_multilingual(args, pt_honkai_star_rail)
    elif args.dataset == "amphoreus":
        entry_multilingual(args, pt_amphoreus)
    elif args.dataset == "vendor":
        entry_vendor(args)
    elif args.dataset == "sft":
        entry_multilingual(args, sft_honkai_star_rail)
    elif args.dataset == "cyrene":
        entry_multilingual(args, sft_cyrene)
    else:
        parser.error(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()
