import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


import argparse
import itertools
import json
import multiprocessing as mp
from pathlib import Path
from typing import IO, Iterator, Literal

from automation.api.conversation import Conversation
from automation.api.paragraph import Paragraph
from automation.api.sentence import Sentence
from automation.api.out_trait import OutTrait
from automation.factories.sft import SftFactory
from automation.factories.pt import PtFactory
from automation.match_sub import MatchSub

import tqdm
from transformers import AutoTokenizer


def get_nickname(language: str):
    return "银河猫猫侠" if language.lower() in ("chs", "cht") else "FelysNeko"


def get_match_sub(language: str, is_male: bool):
    nickname = get_nickname(language)
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


def setup_global_config(language: str, model: Literal["pt", "sft"]):
    auto_format = get_auto_format(language, model)
    tokenizer = AutoTokenizer.from_pretrained(
        "tokenizer", trust_remote_code=True, local_files_only=True
    )
    get_num_tokens = lambda x: len(tokenizer.encode(x))
    match_sub = get_match_sub(language, False)

    Sentence.set_auto_format_func(auto_format)
    Sentence.set_get_num_tokens_func(get_num_tokens)
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


def __split_paragraphs(
    iterable: Iterator[Paragraph], max_token: int
) -> Iterator[Paragraph]:
    for paragraph in iterable:
        yield from paragraph.split(max_token)


def pt_worker(output_path: Path, root_dir: Path, language: str):
    setup_global_config(language, "pt")
    target = PtFactory(root_dir, language).amphoreus.values()
    iterable = itertools.chain.from_iterable(target)

    with open(output_path / f"{language}.jsonl", "w+") as file:
        iterable = __split_paragraphs(iterable, 4096)
        iterable = to_jsonl(iterable, file)
        iterable = num_tokens(iterable)
        metrics = list(tqdm.tqdm(iterable, desc=f"> {language:>3}"))

    return metrics


def __clip_conversations(
    iterable: Iterator[Conversation], max_user_lines: int
) -> Iterator[Conversation]:
    for each in iterable:
        yield each.clip(max_user_lines)


def sft_worker(output_path: Path, root_dir: Path, language: str):
    setup_global_config(language, "sft")
    target = SftFactory(root_dir, language).cyrene.values()
    iterable = itertools.chain.from_iterable(target)

    with open(output_path / f"{language}.jsonl", "w+") as file:
        iterable = __clip_conversations(iterable, 10)
        iterable = to_jsonl(iterable, file)
        iterable = num_tokens(iterable)
        metrics = list(tqdm.tqdm(iterable, desc=f"> {language:>3}"))

        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=Path,
        dest="root_dir",
        required=True,
    )
    parser.add_argument(
        "--mode",
        type=str,
        dest="mode",
        choices=("pt", "sft"),
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("corpora"),
    )
    parser.add_argument(
        "--num-proc",
        dest="num_proc",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    languages = [
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

    output_dir = args.output_dir / args.mode
    os.makedirs(output_dir, exist_ok=True)

    tasks = [(output_dir, args.root_dir, lang) for lang in languages]
    pool = mp.Pool(processes=args.num_proc)

    try:
        match args.mode:
            case "pt":
                metrics = pool.starmap(pt_worker, tasks)
            case "sft":
                metrics = pool.starmap(sft_worker, tasks)
            case _:
                raise

    finally:
        pool.close()
        pool.join()

    named_metrics = {k: v for k, v in zip(languages, metrics)}
    with open(args.output_dir / f"{args.mode}.json", "w+") as f:
        json.dump(named_metrics, f)

    num_tokens = sum(sum(x) for x in named_metrics.values())
    print(f"Estimated number of tokens: {num_tokens}")


if __name__ == "__main__":
    main()
