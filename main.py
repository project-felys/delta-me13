import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


import argparse
import itertools
import json
import multiprocessing as mp
from pathlib import Path
from typing import IO, Iterator, List, Literal, Mapping

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


def get_match_sub(language: str, is_male: bool):
    nickname = "银河猫猫侠" if language.lower() in ("chs", "cht") else "FelysNeko"
    return MatchSub(nickname, is_male)


def get_auto_format(language: str, mode: Literal["pt", "sft"]):
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

    if mode == "pt":
        return pt_auto_format
    elif mode == "sft":
        return sft_auto_format
    else:
        raise


def setup_global_config(language: str, mode: Literal["pt", "sft"]):
    auto_format = get_auto_format(language, mode)
    token_counter = get_token_counter()
    match_sub = get_match_sub(language, False)

    Sentence.set_auto_format_func(auto_format)
    Sentence.set_token_counter_func(token_counter)
    Sentence.set_match_sub_func(match_sub)


def to_jsonl(iterable: Iterator[OutTrait], f: IO[str]) -> Iterator[Paragraph]:
    for each in iterable:
        line = each.to_jsonl()
        json.dump(line, f, ensure_ascii=False)
        print(file=f)
        yield each


def num_tokens(iterable: Iterator[OutTrait]) -> Iterator[int]:
    for each in iterable:
        yield each.num_tokens


def emit(iterable: Iterator[OutTrait], output_path: Path, desc: str):
    with open(output_path, "w+") as file:
        iterable = to_jsonl(iterable, file)
        iterable = num_tokens(iterable)
        metrics = list(tqdm.tqdm(iterable, desc=f"> {desc}"))
    return metrics


def __split_paragraphs(
    iterable: Iterator[Paragraph], max_token: int
) -> Iterator[Paragraph]:
    for paragraph in iterable:
        yield from paragraph.split(max_token)


def __clip_conversations(
    iterable: Iterator[Conversation], max_user_lines: int
) -> Iterator[Conversation]:
    for each in iterable:
        yield each.clip(max_user_lines)


def worker_everything(output_dir: Path, root_dir: Path, language: str):
    setup_global_config(language, "pt")
    target = PtFactory(root_dir, language).everything.values()
    iterable = itertools.chain.from_iterable(target)
    iterable = __split_paragraphs(iterable, 4096)
    return emit(iterable, output_dir / f"{language}.jsonl", f"{language:>3}")


def worker_amphoreus(output_dir: Path, root_dir: Path, language: str):
    setup_global_config(language, "pt")
    target = PtFactory(root_dir, language).amphoreus.values()
    iterable = itertools.chain.from_iterable(target)
    iterable = __split_paragraphs(iterable, 4096)
    return emit(iterable, output_dir / f"{language}.jsonl", f"{language:>3}")


def worker_vendor(output_dir: Path, root_dir: Path):
    setup_global_config("chs", "pt")
    iterable = VendorFactory(root_dir).build_vendor()
    return emit(iterable, output_dir / "vendor.jsonl", "vendor")


def worker_cyrene(output_dir: Path, root_dir: Path, language: str):
    setup_global_config(language, "sft")
    target = SftFactory(root_dir, language).cyrene.values()
    iterable = itertools.chain.from_iterable(target)
    iterable = __clip_conversations(iterable, 10)
    return emit(iterable, output_dir / f"{language}.jsonl", f"{language:>3}")


def __ensure_output_dir(output_dir: Path):
    os.makedirs(output_dir, exist_ok=True)


def __dump_name_metrics(named_metrics: Mapping[str, List[int]], output_path: Path):
    with open(output_path, "w+") as f:
        json.dump(named_metrics, f)

    num_tokens = sum(sum(x) for x in named_metrics.values())
    print(f"Estimated number of tokens: {num_tokens}")


def entry_everything(args: argparse.Namespace):
    output_dir = args.output_dir / "everything"
    __ensure_output_dir(output_dir)

    tasks = [(output_dir, args.root_dir, lang) for lang in LANGUAGES]
    pool = mp.Pool(processes=args.num_proc)

    try:
        metrics = pool.starmap(worker_everything, tasks)
    finally:
        pool.close()
        pool.join()

    named_metrics = {k: v for k, v in zip(LANGUAGES, metrics)}
    __dump_name_metrics(named_metrics, args.output_dir / "everything.json")


def entry_amphoreus(args: argparse.Namespace):
    output_dir = args.output_dir / "amphoreus"
    __ensure_output_dir(output_dir)

    tasks = [(output_dir, args.root_dir, lang) for lang in LANGUAGES]
    pool = mp.Pool(processes=args.num_proc)

    try:
        metrics = pool.starmap(worker_amphoreus, tasks)
    finally:
        pool.close()
        pool.join()

    named_metrics = {k: v for k, v in zip(LANGUAGES, metrics)}
    __dump_name_metrics(named_metrics, args.output_dir / "amphoreus.json")


def entry_vendor(args: argparse.Namespace):
    output_dir = args.output_dir / "vendor"
    __ensure_output_dir(output_dir)

    metrics = worker_vendor(output_dir, args.root_dir)

    named_metrics = {"vendor": metrics}
    __dump_name_metrics(named_metrics, args.output_dir / "vendor.json")


def entry_cyrene(args: argparse.Namespace):
    output_dir = args.output_dir / "cyrene"
    __ensure_output_dir(output_dir)

    tasks = [(output_dir, args.root_dir, lang) for lang in LANGUAGES]
    pool = mp.Pool(processes=args.num_proc)

    try:
        metrics = pool.starmap(worker_cyrene, tasks)
    finally:
        pool.close()
        pool.join()

    named_metrics = {k: v for k, v in zip(LANGUAGES, metrics)}
    __dump_name_metrics(named_metrics, args.output_dir / "cyrene.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["everything", "amphoreus", "vendor", "cyrene"],
        required=True,
    )
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
        "--num-proc",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    if args.dataset == "everything":
        entry_everything(args)
    elif args.dataset == "amphoreus":
        entry_amphoreus(args)
    elif args.dataset == "vendor":
        entry_vendor(args)
    elif args.dataset == "cyrene":
        entry_cyrene(args)
    else:
        parser.error(f"Unsupported dataset: {args.dataset}")

if __name__ == "__main__":
    main()
