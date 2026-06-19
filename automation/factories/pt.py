from collections import defaultdict
import itertools
from pathlib import Path
import re
from typing import Iterable, Mapping, Iterator

import pandas as pd

from automation.api.paragraph import Paragraph
from automation.api.sentence import Sentence
from automation.loaders.turn_based_game_data import TurnBasedGameDataLoader


def str_to_lines(string: str) -> Iterator[str]:
    return (line.strip() for line in re.split(r"\n\u00a0|\n", string))


def lines_to_sentences(lines: Iterator[str]) -> Iterator[Sentence]:
    return (Sentence.plain_text(line) for line in lines if line)


def get_isin_mask(series: pd.Series, keys: Iterable[int] | None = None) -> pd.Series:
    if keys:
        key_set = set(keys)
        return series.isin(key_set)
    else:
        return pd.Series(True, index=series.index)


class PtFactory(TurnBasedGameDataLoader):
    def __init__(
        self, turn_based_game_data_dir: Path, turn_based_game_data_language: str
    ):
        super().__init__(turn_based_game_data_dir, turn_based_game_data_language)

    @property
    def everything(self) -> Mapping[str, Iterator[Paragraph]]:
        return {
            "talk_sentence_config": self.build_talk_sentence_config(),
            "book_series_config": self.build_book_series_config(),
            "story_atlas": self.build_story_atlas(),
            "voice_atlas": self.build_voice_atlas(),
            "noun_atlas": self.build_noun_atlas(),
            "chronicle_conclusion": self.build_chronicle_conclusion(),
            "tarot_mails": self.build_tarot_mails(),
            "tarot_book_sentence": self.build_tarot_book_sentence(),
            "tarot_wiki_data": self.build_tarot_wiki_data(),
        }

    @property
    def amphoreus(self) -> Mapping[str, Iterator[Paragraph]]:
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
        extra_name_hashes = [
            11001695012879251917,  # 迷迷
            16450988707457331866,  # 来古士
        ]

        return {
            "talk_sentence_config": self.build_talk_sentence_config(
                list(avatar_id_to_name_hash.values()) + extra_name_hashes
            ),
            "chronicle_conclusion": self.build_chronicle_conclusion(
                [
                    104,  # 主线
                    204,  # 支线
                    803,  # 活动
                ]
            ),
            "story_atlas": self.build_story_atlas(avatar_id_to_name_hash.keys()),
            "voice_atlas": self.build_voice_atlas(avatar_id_to_name_hash.keys()),
            "book_series_config": self.build_book_series_config(
                [
                    5,  # 翁法罗斯
                ]
            ),
            "noun_atlas": self.build_noun_atlas(),  # 智库（专有名词、星神、派系）
            "tarot_mails": self.build_tarot_mails(),  # 邮箱.exe
            "tarot_book_sentence": self.build_tarot_book_sentence(),  # 翁法罗斯英雄纪.exe
            "tarot_wiki_data": self.build_tarot_wiki_data(),  # δ-me13.exe
        }

    def build_talk_sentence_config(self, name_hashes: Iterable[int] | None = None):
        df = self.talk_sentence_config_table
        name_hash_mask = get_isin_mask(df["textmap_talk_sentence_name"], name_hashes)
        group_ids = set(df.loc[name_hash_mask, "group"])
        mask = df["group"].isin(group_ids)
        df = df[mask]

        for _, sub_df in df.groupby("group"):
            sentences = (
                self.sentence_factory(talk_sentence_id, name_hash, text_hash)
                for talk_sentence_id, _, name_hash, text_hash, _ in sub_df.itertuples(
                    index=False
                )
            )
            yield Paragraph(sentences=tuple(sentences))

    def build_book_series_config(
        self, book_series_worlds: Iterable[int] | None = None
    ) -> Iterator[Paragraph]:
        df = self.book_series_config_table
        mask = get_isin_mask(df["book_series_world"], book_series_worlds)
        df = df[mask]

        for (
            book_series_id,
            book_series_hash,
            book_series_comments_hash,
            _,
        ) in df.itertuples(index=False):
            local_books = self.local_book_config_table
            sub_df = local_books[local_books["book_series_id"] == book_series_id]
            for _, book_inside_name_hash, book_content_hash in sub_df.itertuples(
                index=False
            ):
                content = self.get_joined_text_unwrap(book_content_hash)
                lines = itertools.chain(
                    [
                        self.get_joined_text_unwrap(book_series_hash),
                        self.get_joined_text(book_series_comments_hash, ""),
                        self.get_joined_text_unwrap(book_inside_name_hash),
                    ],
                    str_to_lines(content),
                )
                sentences = lines_to_sentences(lines)
                yield Paragraph(sentences=tuple(sentences))

    def build_voice_atlas(
        self, avatar_ids: Iterable[int] | None = None
    ) -> Iterator[Paragraph]:
        df = self.voice_atlas_table
        mask = get_isin_mask(df["avatar_id"], avatar_ids)
        df = df.loc[mask, ["avatar_id", "voice_title_hash", "voice_m_hash", "sort_id"]]

        for avatar_id, sub_df in df.groupby("avatar_id"):
            name = self.avatar_id_to_name_map[avatar_id]
            sentences = []
            sorted_sub_df = sub_df.sort_values("sort_id")
            for _, voice_title_hash, voice_m_hash, _ in sorted_sub_df.itertuples(
                index=False
            ):
                voice_title = self.get_joined_text_unwrap(voice_title_hash)
                voice_m = self.get_joined_text_unwrap(voice_m_hash)
                sentences.append(Sentence.talk_sentence("{NICKNAME}", voice_title))
                sentences.append(Sentence.talk_sentence(name, voice_m))
            yield Paragraph(sentences=tuple(sentences))

    def build_story_atlas(
        self, avatar_ids: Iterable[int] | None = None
    ) -> Iterator[Paragraph]:
        df = self.story_atlas_table
        mask = get_isin_mask(df["avatar_id"], avatar_ids)
        df = df[mask]

        for avatar_id, sub_df in df.groupby("avatar_id"):
            name = self.avatar_id_to_name_map[avatar_id]

            id_to_story_text = {}
            for _, story_id, story_hash, replace_id in sub_df.itertuples(index=False):
                text = self.get_joined_text_unwrap(story_hash)

                if replace_id is pd.NA:
                    id_to_story_text[story_id] = text
                else:
                    id_to_story_text[replace_id] = text

            lines = [name]
            for _, story_text in sorted(id_to_story_text.items()):
                story_lines = str_to_lines(story_text)
                lines.extend(story_lines)

            sentences = lines_to_sentences(lines)
            yield Paragraph(sentences=tuple(sentences))

    def build_noun_atlas(self) -> Iterator[Paragraph]:
        for (
            noun_title_hash,
            noun_desc_hash,
        ) in self.none_atlas_table.itertuples(index=False):
            content = self.get_joined_text_unwrap(noun_desc_hash)
            lines = itertools.chain(
                [self.get_joined_text_unwrap(noun_title_hash)],
                str_to_lines(content),
            )
            sentences = lines_to_sentences(lines)
            yield Paragraph(sentences=tuple(sentences))

    def build_chronicle_conclusion(
        self, first_3_digits: Iterable[int] | None = None
    ) -> Iterator[Paragraph]:
        df = self.chronicle_conclusion_table
        series = df["mission_id"] // 10000
        mask = get_isin_mask(series, first_3_digits)
        df = df[mask]

        groups = defaultdict(list)
        for (
            sub_mission_id,
            *rest,
        ) in self.sub_mission_table.itertuples(index=False):
            groups[sub_mission_id // 100].append(rest)

        for (
            mission_id,
            mission_conclusion_hash,
        ) in df.itertuples(index=False):
            content = self.get_joined_text_unwrap(mission_conclusion_hash)

            lines = [
                self.main_mission_id_to_name_map[mission_id],
                *str_to_lines(content),
            ]

            prev = None
            for target_text_hash, descrption_text_hash in groups[mission_id]:
                target_text = self.get_joined_text(target_text_hash, "")
                descrption_text = self.get_joined_text(descrption_text_hash, "")

                current = (target_text, descrption_text)
                if not (prev != current and target_text and descrption_text):
                    continue
                prev = current

                lines.append(target_text)
                lines.extend(str_to_lines(descrption_text))

            sentences = lines_to_sentences(lines)
            yield Paragraph(sentences=tuple(sentences))

    def build_tarot_mails(self) -> Iterator[Paragraph]:
        for (
            title_hash,
            from_hash,
            to_hash,
            mail_sentence_id_list,
        ) in self.tarot_mailbox_table.itertuples(index=False):
            lines = itertools.chain(
                [
                    self.get_joined_text_unwrap(title_hash),
                    self.get_joined_text_unwrap(from_hash),
                    self.get_joined_text_unwrap(to_hash),
                ],
                (
                    self.mail_sentence_id_to_sentence_map[mail_sentence_id]
                    for mail_sentence_id in mail_sentence_id_list
                ),
            )
            sentences = lines_to_sentences(lines)
            yield Paragraph(sentences=tuple(sentences))

    def build_tarot_book_sentence(self) -> Iterator[Paragraph]:
        groups = defaultdict(list)
        for id, sentence_hash in self.tarot_book_sentence_table.itertuples(index=False):
            groups[id // 1000].append(sentence_hash)

        for group in groups.values():
            lines = []
            for sentence_hash in group:
                text = self.get_joined_text_unwrap(sentence_hash)
                lines.extend(str_to_lines(text))
            sentences = lines_to_sentences(lines)
            yield Paragraph(sentences=tuple(sentences))

    def build_tarot_wiki_data(self) -> Iterator[Paragraph]:
        groups = defaultdict(list)
        for id, *rest in self.tarot_wiki_data_table.itertuples(index=False):
            groups[id // 100].append(rest)

        for group in groups.values():
            lines = []

            for title_hash, details_hash, subdata_list in group:
                title = self.get_joined_text_unwrap(title_hash)
                details = self.get_joined_text_unwrap(details_hash)
                lines.append(title)
                lines.extend(str_to_lines(details))

                for subdata_id in subdata_list:
                    sub_title_hash, sub_details_hash = (
                        self.tarot_wiki_subdata_table.loc[subdata_id]
                    )
                    sub_title = self.get_joined_text_unwrap(sub_title_hash)
                    sub_details = self.get_joined_text_unwrap(sub_details_hash)
                    lines.append(sub_title)
                    lines.extend(str_to_lines(sub_details))

            sentences = lines_to_sentences(lines)
            yield Paragraph(sentences=tuple(sentences))
