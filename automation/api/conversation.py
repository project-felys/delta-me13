from typing import Any, Iterator, Mapping, Tuple

from pydantic import BaseModel, ConfigDict

from automation.api.out_trait import OutTrait
from automation.api.sentence import Sentence


class Round(BaseModel):
    model_config = ConfigDict(frozen=True)

    user: Tuple[Sentence, ...]
    assistant: Tuple[Sentence, ...]

    @property
    def num_tokens(self):
        user_tokens = sum(s.num_tokens for s in self.user)
        assistant_tokens = sum(s.num_tokens for s in self.assistant)
        return user_tokens + assistant_tokens

    def clip(self, max_user_lines: int) -> "Round":
        return Round(user=self.user[-max_user_lines:], assistant=self.assistant)

    @classmethod
    def single_line(cls, user: str, assistant: str) -> "Round":
        return cls(
            user=tuple([Sentence.plain_text(user)]),
            assistant=tuple([Sentence.plain_text(assistant)]),
        )

    def to_jsonl(self) -> Iterator[Mapping[str, str]]:
        user_content = "\n".join(s.pretty_string for s in self.user)
        if user_content:
            yield {"role": "user", "content": user_content}
        assistant_content = "\n".join(s.pretty_string for s in self.assistant)
        yield {"role": "assistant", "content": assistant_content}


class Conversation(BaseModel, OutTrait):
    model_config = ConfigDict(frozen=True)

    rounds: Tuple[Round, ...]

    @property
    def num_tokens(self) -> int:
        return sum(x.num_tokens for x in self.rounds)

    @classmethod
    def single_round(cls, user: str, assistant: str) -> "Conversation":
        return cls(rounds=tuple([Round.single_line(user, assistant)]))

    def clip(self, max_user_lines: int) -> "Conversation":
        return Conversation(rounds=tuple(x.clip(max_user_lines) for x in self.rounds))

    def to_jsonl(self) -> Mapping[str, Any]:
        messages = list(line for round in self.rounds for line in round.to_jsonl())
        return {"messages": messages}
