import re
from pathlib import Path
from typing import Iterator

from automation.api.paragraph import Paragraph
from automation.api.sentence import Sentence
from automation.loaders.vendor import VendorLoader


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


class VendorFactory(VendorLoader):
    def __init__(self, vendor_dir: Path):
        super().__init__(vendor_dir)

    def build_miyoushe(self) -> Iterator[Paragraph]:
        for text in self.miyoushe.values():
            lines = str_to_lines(text)
            sentences = lines_to_sentences(lines)
            yield Paragraph(sentences=tuple(sentences))

    def build_coig(self) -> Iterator[Paragraph]:
        for instruction, input_text, output_text, _, _ in self.coig.itertuples(
            index=False
        ):
            lines = (
                instruction,
                unescape_string(input_text),
                unescape_string(output_text),
            )
            sentences = lines_to_sentences(lines)
            yield Paragraph(sentences=tuple(sentences))
