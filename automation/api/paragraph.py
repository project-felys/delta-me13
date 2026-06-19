from typing import Any, Iterator, Mapping, Tuple

from pydantic import BaseModel, ConfigDict

from automation.api.out_trait import OutTrait
from automation.api.sentence import Sentence


class Paragraph(BaseModel, OutTrait):
    model_config = ConfigDict(frozen=True)

    sentences: Tuple[Sentence, ...]

    @property
    def num_tokens(self) -> int:
        return sum(s.num_tokens for s in self.sentences)

    def split(self, max_tokens: int) -> Iterator[Paragraph]:
        current_num_tokens = 0
        start = 0

        for i, sentence in enumerate(self.sentences):
            num_tokens = sentence.num_tokens
            if current_num_tokens + num_tokens > max_tokens and start < i:
                yield Paragraph(sentences=self.sentences[start:i])
                start = i
                current_num_tokens = num_tokens
            else:
                current_num_tokens += num_tokens

        if start < len(self.sentences):
            yield Paragraph(sentences=self.sentences[start:])

    def to_jsonl(self, use_system: bool) -> Mapping[str, Any]:
        assert not use_system
        lines = (s.pretty_string for s in self.sentences if s.pretty_string)
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": "\n".join(lines),
                }
            ]
        }
