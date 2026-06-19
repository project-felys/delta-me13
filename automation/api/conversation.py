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
    def single_line(
        cls, user_text: str, assistant_name: str, assistant_text: str
    ) -> "Round":
        return cls(
            user=tuple([Sentence.plain_text(user_text)]),
            assistant=tuple(
                [
                    Sentence(
                        talk_sentence_id=None,
                        name=assistant_name,
                        name_hash=None,
                        text=assistant_text,
                        text_hash=None,
                    )
                ]
            ),
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
    def single_round(
        cls, user_text: str, assistant_name: str, assistant_text: str
    ) -> "Conversation":
        return cls(
            rounds=tuple([Round.single_line(user_text, assistant_name, assistant_text)])
        )

    def split(self, max_tokens: int) -> Iterator[Conversation]:
        current_num_tokens = 0
        start = 0

        for i, round in enumerate(self.rounds):
            num_tokens = round.num_tokens
            if current_num_tokens + num_tokens > max_tokens and start < i:
                yield Conversation(rounds=self.rounds[start:i])
                start = i
                current_num_tokens = num_tokens
            else:
                current_num_tokens += num_tokens

        if start < len(self.rounds):
            yield Conversation(rounds=self.rounds[start:])

    def clip(self, max_user_lines: int) -> "Conversation":
        return Conversation(rounds=tuple(x.clip(max_user_lines) for x in self.rounds))

    def to_jsonl(self, use_system: bool) -> Mapping[str, Any]:
        name = self.rounds[0].assistant[0].pretty_name
        if use_system:
            assert name
        else:
            name = ""
        messages = [{"role": "system", "content": name}]
        messages.extend(line for round in self.rounds for line in round.to_jsonl())
        return {"messages": messages}
