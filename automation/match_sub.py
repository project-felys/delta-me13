import re
from typing import Set, Tuple


class MatchSub:
    _remove_parts: Tuple[str, ...] = (
        r"<[^>]+>",
        r"\{RUBY_B#[^}]*\}",
        r"\{RUBY_E#\}",
        r"\{F#\{M#\}\}",
        r"\{Img(#| )\d+\}",
        r"\{LAYOUT_CONTROLLER#[^}]+\}",
        r"\{LAYOUT_KEYBOARD#[^}]+\}",
    )
    _heliobi_count_str: str = "{MCV#8015162#OldValue_1}"
    _space_str: str = "{SPACE}"
    _nickname_str: str = "{NICKNAME}"

    _remove_pattern: re.Pattern[str] = re.compile("|".join(_remove_parts))
    _layout_mobile_pattern: re.Pattern[str] = re.compile(r"\{LAYOUT_MOBILE#([^}]*)\}")
    _male_pattern: re.Pattern[str] = re.compile(r"\{M#([^}]*)\}")
    _female_pattern: re.Pattern[str] = re.compile(r"\{F#([^}]*)\}")
    _warning_pattern: re.Pattern[str] = re.compile(r"\{[^}]*\}")

    _skip_warning: Set[str] = {"{大地獣, ネクタール, 数学}"}

    def __init__(self, nickname: str, is_male: bool) -> None:
        self.nickname = nickname
        self.is_male = is_male

    def __call__(self, s: str) -> str:
        s = s.replace(self._nickname_str, self.nickname)
        s = s.replace(self._heliobi_count_str, "3")
        s = s.replace(self._space_str, " ")

        s = self._remove_pattern.sub("", s)
        s = self._layout_mobile_pattern.sub(r"\1", s)

        if self.is_male:
            s = self._male_pattern.sub(r"\1", s)
            s = self._female_pattern.sub("", s)
        else:
            s = self._female_pattern.sub(r"\1", s)
            s = self._male_pattern.sub("", s)

        raw_warning = self._warning_pattern.findall(s)
        warning = [w for w in raw_warning if w not in self._skip_warning]
        if warning:
            print(warning)

        return s
