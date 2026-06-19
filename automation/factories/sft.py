import itertools
from pathlib import Path
from typing import Iterator, Mapping

import pandas as pd

from automation.api.conversation import Conversation, Round
from automation.turn_based_game_loader import TurnBasedGameLoader


class SftFactory(TurnBasedGameLoader):
    def __init__(self, root_dir: Path, language: str):
        super().__init__(root_dir, language)

    @property
    def everything(self) -> Mapping[str, Iterator[Conversation]]:
        name_hashes = (
            self.talk_sentence_config_table["textmap_talk_sentence_name"]
            .dropna()
            .unique()
        )
        avatar_ids = self.avatar_id_to_name_map.keys()

        talk_sentence_config_list = [
            self.build_talk_sentence_config(name_hash) for name_hash in name_hashes
        ]
        voice_atlas_list = [
            self.build_voice_atlas(avatar_id) for avatar_id in avatar_ids
        ]

        return {
            "talk_sentence_config": itertools.chain(*talk_sentence_config_list),
            "voice_atlas": itertools.chain(*voice_atlas_list),
        }

    @property
    def cyrene(self) -> Mapping[str, Iterator[Conversation]]:
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
        talk_sentence_config_list = [
            self.build_talk_sentence_config(name_hash) for name_hash in name_hashes
        ]

        return {
            "talk_sentence_config": itertools.chain(*talk_sentence_config_list),
            "voice_atlas": self.build_voice_atlas(1415),  # 昔涟
        }

    def build_talk_sentence_config(self, name_hash: int) -> Iterator[Conversation]:
        df = self.talk_sentence_config_table
        name_hash_mask = df["textmap_talk_sentence_name"] == name_hash
        group_ids = set(df.loc[name_hash_mask, "group"])
        mask = df["group"].isin(group_ids)
        df = df[mask]

        for _, sub_df in df.groupby("group"):
            yield self.__build_one_talk_sentence_config(sub_df, name_hash)

    def __build_one_talk_sentence_config(
        self, df: pd.DataFrame, assistant_name_hash: int
    ) -> Conversation:
        df = df[
            ["talk_sentence_id", "textmap_talk_sentence_name", "talk_sentence_text"]
        ]
        rounds = []

        i = 0
        while i < len(df):
            user = []
            while i < len(df):
                talk_sentence_id, name_hash, text_hash = df.iloc[i]
                if name_hash == assistant_name_hash:
                    break
                sentence = self.sentence_factory(talk_sentence_id, name_hash, text_hash)
                user.append(sentence)
                i += 1

            assistant = []
            while i < len(df):
                talk_sentence_id, name_hash, text_hash = df.iloc[i]
                if name_hash != assistant_name_hash:
                    break
                sentence = self.sentence_factory(talk_sentence_id, name_hash, text_hash)
                assistant.append(sentence)
                i += 1

            if assistant:
                round = Round(user=tuple(user), assistant=tuple(assistant))
                rounds.append(round)

        return Conversation(rounds=rounds)

    def build_voice_atlas(self, avatar_id: int) -> Iterator[Conversation]:
        mask = self.voice_atlas_table["avatar_id"] == avatar_id
        name = self.avatar_id_to_name_map[avatar_id]
        df = self.voice_atlas_table[mask]
        for _, voice_title_hash, voice_m_hash, _ in df.itertuples(index=False):
            voice_title = self.get_joined_text_unwrap(voice_title_hash)
            voice_m = self.get_joined_text_unwrap(voice_m_hash)
            yield Conversation.single_round(voice_title, name, voice_m)
