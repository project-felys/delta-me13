import abc
from typing import Mapping, Any


class OutTrait(abc.ABC):
    @property
    @abc.abstractmethod
    def num_tokens(self) -> int: ...

    @abc.abstractmethod
    def to_jsonl(self) -> Mapping[str, Any]: ...
