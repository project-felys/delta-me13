from automation.match_sub import MatchSub


def get_felysneko_match_sub(language: str) -> MatchSub:
    nickname = "银河猫猫侠" if language.lower() in ["chs", "cht"] else "FelysNeko"
    return MatchSub(nickname=nickname, slow_path=True, enable_warning=True)


def get_fast_felysneko_match_sub() -> MatchSub:
    return MatchSub(nickname="银河猫猫侠", slow_path=False, enable_warning=False)


def get_slow_path_only_match_sub() -> MatchSub:
    return MatchSub(nickname="{NICKNAME}", slow_path=True, enable_warning=False)
