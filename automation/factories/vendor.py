import functools
import os
from pathlib import Path
from typing import Iterator, Mapping

from automation.api.paragraph import Paragraph
from automation.api.sentence import Sentence


def str_to_lines(string: str) -> Iterator[str]:
    return (line.strip() for line in string.split())


def lines_to_sentences(lines: Iterator[str]) -> Iterator[Sentence]:
    return (Sentence.plain_text(line) for line in lines if line)


class VendorFactory:
    def __init__(self, root_dir: Path):
        self.__root_dat = root_dir

    @functools.cached_property
    def __vendor(self) -> Mapping[str, str]:
        vendor = {}
        for file_name in os.listdir(self.__root_dat):
            file_path = self.__root_dat / file_name
            if not file_path.is_file():
                continue
            with open(file_path) as file:
                vendor[file_name] = file.read()
        return vendor

    def build_vendor(self) -> Iterator[Paragraph]:
        for text in self.__vendor.values():
            lines = str_to_lines(text)
            sentences = lines_to_sentences(lines)
            yield Paragraph(sentences=tuple(sentences))
