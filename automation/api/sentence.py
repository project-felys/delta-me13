import functools
from typing import Callable

from pydantic import BaseModel, ConfigDict


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
    def pretty_string(self) -> str:
        s = self.__auto_format(self)
        return self.__match_sub_func(s)

    @functools.cached_property
    def num_tokens(self) -> int:
        s = self.pretty_string
        return self.__get_token_counter_func(s)

    @staticmethod
    def __auto_format(sentence: Sentence) -> str:
        return sentence.text

    @classmethod
    def set_auto_format_func(cls, backend: Callable[[Sentence], str]) -> None:
        cls.__auto_format = staticmethod(backend)

    @staticmethod
    def __match_sub_func(s: str) -> str:
        return s

    @classmethod
    def set_match_sub_func(cls, backend: Callable[[str], str]) -> None:
        cls.__match_sub_func = staticmethod(backend)

    @staticmethod
    def __get_token_counter_func(s: str) -> int:
        return len(s)

    @classmethod
    def set_token_counter_func(cls, backend: Callable[[str], int]) -> None:
        cls.__get_token_counter_func = staticmethod(backend)
