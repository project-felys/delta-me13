import itertools
from pathlib import Path
from typing import Iterator, List

from automation.api.conversation import Conversation
from automation.api.sentence import Sentence
from automation.factories.sft import SftFactory
from automation.implementation.auto_format import (
    get_cyrene_whitelist_auto_format,
)
from automation.implementation.match_sub import get_felysneko_match_sub
from automation.implementation.task.out_trait import emit
from automation.implementation.token_counter import get_qwen3_token_counter


def __clip_conversations(
    iterable: Iterator[Conversation], max_user_lines: int
) -> Iterator[Conversation]:
    for each in iterable:
        yield each.clip(max_user_lines)


def __split_conversations(
    iterable: Iterator[Conversation], max_token: int
) -> Iterator[Conversation]:
    for each in iterable:
        yield from each.split(max_token)


def __filter_conversations(
    iterable: Iterator[Conversation], max_token: int
) -> Iterator[Conversation]:
    for each in iterable:
        if each.num_tokens > max_token or each.is_self_talk():
            continue
        yield each


def everything(
    output_dir: Path, turn_based_game_data_dir: Path, language: str
) -> List[int]:
    Sentence.global_config(
        auto_format=None,
        token_counter=get_qwen3_token_counter(),
        match_sub=get_felysneko_match_sub(language),
    )

    factory = SftFactory(turn_based_game_data_dir, language)
    name_hash_set = set(
        factory.talk_sentence_config_table["textmap_talk_sentence_name"]
    )
    voice_atlas_list = [
        factory.build_voice_atlas(avatar_id)
        for avatar_id in factory.avatar_id_to_name_map.keys()
    ]
    iterable = itertools.chain(
        factory.build_talk_sentence_config_everyone_in_group(name_hash_set),
        itertools.chain(*voice_atlas_list),
    )
    iterable = __clip_conversations(iterable, 10)
    iterable = __split_conversations(iterable, 4096)
    iterable = __filter_conversations(iterable, 4096)
    return emit(
        iterable, output_dir / f"{language}.jsonl", f"{language:>3}", use_system=True
    )


def amphoreus(
    output_dir: Path, turn_based_game_data_dir: Path, language: str
) -> List[int]:
    Sentence.global_config(
        auto_format=None,
        token_counter=get_qwen3_token_counter(),
        match_sub=get_felysneko_match_sub(language),
    )

    avatar_id_to_name_hash = {
        1402: 8347254212154585286,  # 阿格莱雅
        1403: 12286569378821401368,  # 缇宝
        1404: 12172129776731566058,  # 万敌
        1405: 7940279852062189396,  # 那刻夏
        1406: 8212064977546372217,  # 赛飞儿
        1407: 3884071463804277504,  # 遐蝶
        1408: 1410515507232658695,  # 白厄
        1409: 11373702895576004432,  # 风堇
        1410: 6101302014640441508,  # 海瑟音
        1412: 16138667287721516920,  # 刻律德菈
        1413: 6287657147795310895,  # 长夜月
        1414: 11464334189321098131,  # 丹恒
        1415: 2309313067306506373,  # 昔涟
    }
    extra_name_hashes = {
        11001695012879251917,  # 迷迷
        16450988707457331866,  # 来古士
    }

    factory = SftFactory(turn_based_game_data_dir, language)
    name_hash_set = set(avatar_id_to_name_hash.values()) | extra_name_hashes
    voice_atlas_list = [
        factory.build_voice_atlas(avatar_id)
        for avatar_id in avatar_id_to_name_hash.keys()
    ]
    iterable = itertools.chain(
        factory.build_talk_sentence_config_everyone_in_group(name_hash_set),
        itertools.chain(*voice_atlas_list),
    )
    iterable = __clip_conversations(iterable, 10)
    iterable = __split_conversations(iterable, 4096)
    iterable = __filter_conversations(iterable, 4096)
    return emit(
        iterable, output_dir / f"{language}.jsonl", f"{language:>3}", use_system=True
    )


def cyrene(
    output_dir: Path, turn_based_game_data_dir: Path, language: str
) -> List[int]:
    Sentence.global_config(
        auto_format=get_cyrene_whitelist_auto_format(language),
        token_counter=get_qwen3_token_counter(),
        match_sub=get_felysneko_match_sub(language),
    )

    name_hashes = [
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
    ]

    factory = SftFactory(turn_based_game_data_dir, language)
    talk_sentence_config_list = [
        factory.build_talk_sentence_config_single_character(name_hash)
        for name_hash in name_hashes
    ]
    iterable = itertools.chain(
        itertools.chain(*talk_sentence_config_list),
        factory.build_voice_atlas(1415),  # 昔涟
    )
    iterable = __clip_conversations(iterable, 10)
    iterable = __filter_conversations(iterable, 8192)
    return emit(
        iterable, output_dir / f"{language}.jsonl", f"{language:>3}", use_system=False
    )
