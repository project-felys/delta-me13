from typing import Callable

from automation.api.sentence import Sentence


def get_retain_sentence_name_auto_format(
    language: str,
) -> Callable[[Sentence], str]:
    colon = ": "
    if language.lower() in ["chs", "cht"]:
        colon = "："

    def retain_sentence_name(sentence: Sentence) -> str:
        text = sentence.text
        if sentence.name is None:
            return text
        else:
            return f"{sentence.name}{colon}{text}"

    return retain_sentence_name


def get_cyrene_whitelist_auto_format(language: str) -> Callable[[Sentence], str]:
    lparen = "("
    rparen = ")"
    if language.lower() in ["chs", "cht"]:
        lparen = "（"
        rparen = "）"

    def cyrene_whitelist(sentence: Sentence) -> str:
        text = sentence.text.lstrip(lparen).rstrip(rparen)
        name_hash = sentence.name_hash
        if (
            name_hash is None
            or name_hash == 7108701208745180573  # {NICKNAME}
            or name_hash
            in {
                2309313067306506373,  # 昔涟
                11001695012879251917,  # 迷迷
                6462539533232001276,  # 少女的声音
                402745232924048698,  # 「记忆的花」
                14378608167795068022,  # 「记忆的花蕾」
                122935082492335341,  # 「记忆的幼芽」
                11263148736238834705,  # 「记忆的种子」
                6846009442988187041,  # {NICKNAME}&昔涟
                4455374036049186635,  # 昔涟
                13483973725572441939,  # 「另一位作者♪」
                12560857117710338840,  # 往昔的涟漪
            }
        ):
            return text
        else:
            return f"{lparen}{text}{rparen}"

    return cyrene_whitelist
