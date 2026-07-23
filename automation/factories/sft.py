from pathlib import Path
from typing import Iterator, Set

import pandas as pd

from automation.api.conversation import Conversation, Round
from automation.loaders.turn_based_game_data import TurnBasedGameDataLoader


class SftFactory(TurnBasedGameDataLoader):
    def __init__(
        self, turn_based_game_data_dir: Path, turn_based_game_data_language: str
    ):
        super().__init__(turn_based_game_data_dir, turn_based_game_data_language)

    def build_talk_sentence_config_single_character(
        self, name_hash: int
    ) -> Iterator[Conversation]:
        df = self.talk_sentence_config_table.fillna(None)
        name_hash_mask = df["textmap_talk_sentence_name"] == name_hash
        group_ids = set(df.loc[name_hash_mask, "group"])
        mask = df["group"].isin(group_ids)
        df = df[mask]

        for _, sub_df in df.groupby("group"):
            yield self.__build_one_talk_sentence_config(sub_df, name_hash)

    def build_talk_sentence_config_everyone_in_group(
        self, name_hash_set: Set[int]
    ) -> Iterator[Conversation]:
        df = self.talk_sentence_config_table.fillna(None)
        target_sub_dfs = (
            sub_df
            for _, sub_df in df.groupby("group")
            if sub_df["textmap_talk_sentence_name"].isin(name_hash_set).any()
        )

        for sub_df in target_sub_dfs:
            for name_hash in sub_df["textmap_talk_sentence_name"].unique():
                yield self.__build_one_talk_sentence_config(sub_df, name_hash)

    def __build_one_talk_sentence_config(
        self, df: pd.DataFrame, assistant_name_hash: int
    ) -> Conversation:
        fields = [
            "talk_sentence_id",
            "textmap_talk_sentence_name",
            "talk_sentence_text",
        ]
        df = df[fields]
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
        df = self.voice_atlas_table
        mask = df["avatar_id"] == avatar_id
        df = df.loc[mask, ["voice_title_hash", "voice_m_hash"]]

        name = self.avatar_id_to_name_map[avatar_id]
        for voice_title_hash, voice_m_hash in df.itertuples(index=False):
            voice_title = self.get_joined_text_unwrap(voice_title_hash)
            voice_m = self.get_joined_text_unwrap(voice_m_hash)
            yield Conversation.single_round(voice_title, name, voice_m)
