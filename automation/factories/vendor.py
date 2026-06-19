import functools
import json
import os
import re
from pathlib import Path
from typing import Iterator, Mapping
import pandas as pd

from automation.api.paragraph import Paragraph
from automation.api.sentence import Sentence


def unescape_string(s):
    escape_map = {
        "\\n": "\n",
        "\\t": "\t",
        "\\r": "\r",
        "\\f": "\f",
        "\\b": "\b",
        "\\'": "'",
        '\\"': '"',
        "\\\\": "\\",
    }
    pattern = re.compile("|".join(re.escape(k) for k in escape_map.keys()))
    return pattern.sub(lambda m: escape_map[m.group()], s)


def str_to_lines(string: str) -> Iterator[str]:
    return (line.strip() for line in string.split())


def lines_to_sentences(lines: Iterator[str]) -> Iterator[Sentence]:
    return (Sentence.plain_text(line) for line in lines if line)


class VendorFactory:
    def __init__(self, root_dir: Path):
        self.__root_dir = root_dir

    def build_miyoushe(self) -> Iterator[Paragraph]:
        for text in self.__miyoushe.values():
            lines = str_to_lines(text)
            sentences = lines_to_sentences(lines)
            yield Paragraph(sentences=tuple(sentences))

    def build_coig(self) -> Iterator[Paragraph]:
        for instruction, input_text, output_text, _, _ in self.__coig.itertuples(
            index=False
        ):
            lines = (
                instruction,
                unescape_string(input_text),
                unescape_string(output_text),
            )
            sentences = lines_to_sentences(lines)
            yield Paragraph(sentences=tuple(sentences))

    @functools.cached_property
    def __miyoushe(self) -> Mapping[str, str]:
        path_to_miyoushe = self.__root_dir / "miyoushe"
        name_to_description = {}
        for file_name in os.listdir(path_to_miyoushe):
            file_path = path_to_miyoushe / file_name
            if not file_path.is_file():
                continue
            with open(file_path) as file:
                name_to_description[file_name] = file.read()
        return name_to_description

    @functools.cached_property
    def __coig(self) -> pd.DataFrame:
        with open(self.__root_dir / "coig" / "leetcode_instructions.jsonl") as f:
            data = (json.loads(line) for line in f.readlines())
            return pd.DataFrame(data=data)
