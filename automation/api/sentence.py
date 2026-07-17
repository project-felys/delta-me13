import functools
from typing import Callable

from pydantic import BaseModel, ConfigDict


def default_auto_format(sentence: Sentence) -> str:
    return sentence.text


def default_match_sub(s: str) -> str:
    return s


def default_token_counter(s: str) -> int:
    return len(s)


class Sentence(BaseModel):
    model_config = ConfigDict(frozen=True)

    talk_sentence_id: int | None
    name: str | None
    name_hash: int | None
    text: str
    text_hash: int | None

    @classmethod
    def plain_text(cls, text: str) -> "Sentence":
        return cls(
            talk_sentence_id=None, name=None, name_hash=None, text=text, text_hash=None
        )

    @classmethod
    def talk_sentence(cls, name: str, text: str) -> "Sentence":
        return cls(
            talk_sentence_id=None, name=name, name_hash=None, text=text, text_hash=None
        )

    @functools.cached_property
    def pretty_name(self) -> str:
        return self.__match_sub_func(self.name)

    @functools.cached_property
    def pretty_string(self) -> str:
        s = self.__auto_format(self)
        return self.__match_sub_func(s)

    @functools.cached_property
    def num_tokens(self) -> int:
        s = self.pretty_string
        return self.__get_token_counter_func(s)

    __auto_format = staticmethod(default_auto_format)
    __match_sub_func = staticmethod(default_match_sub)
    __get_token_counter_func = staticmethod(default_token_counter)

    @classmethod
    def global_config(
        cls,
        *,
        auto_format: Callable[[Sentence], str] | None,
        match_sub: Callable[[str], str] | None,
        token_counter: Callable[[str], int] | None,
    ) -> None:
        cls.__auto_format = staticmethod(auto_format or default_auto_format)
        cls.__match_sub_func = staticmethod(match_sub or default_match_sub)
        cls.__get_token_counter_func = staticmethod(
            token_counter or default_token_counter
        )
