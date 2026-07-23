import functools
from pathlib import Path
from typing import Iterable, Iterator, List, Pattern, Tuple

import pandas as pd

from automation.api.audio import Audio
from automation.api.sentence import Sentence
from automation.fnv1 import fnv1_32, fnv1_64
from automation.loaders.unpacked_audio_language import UnpackedAudioLanguageLoader
from automation.loaders.turn_based_game_data import TurnBasedGameDataLoader

LANGUAGE_ABBREVIATION_MAP = {
    "Chinese(PRC)": "chs",
    "English": "en",
    "Japanese": "jp",
    "Korean": "kr",
}


class TtsFactory(UnpackedAudioLanguageLoader, TurnBasedGameDataLoader):
    def __init__(
        self,
        unpacked_audio_language_dir: Path,
        turn_based_game_data_dir: Path,
    ):
        language = unpacked_audio_language_dir.name
        UnpackedAudioLanguageLoader.__init__(self, unpacked_audio_language_dir)
        TurnBasedGameDataLoader.__init__(
            self, turn_based_game_data_dir, LANGUAGE_ABBREVIATION_MAP[language]
        )
        self.__language = language

    def voice_path_to_id(self, voice_path: str) -> int:
        wem_path = f"{self.__language}/voice/{voice_path}.wem"
        return fnv1_64(wem_path.lower())

    def audio_event_to_id(self, audio_event: str) -> int:
        event_id = fnv1_32(audio_event.lower())
        return self.event_id_to_first_bank_map[event_id]

    def text_hash_to_sentence(self, text_hash: int) -> Sentence:
        text = self.get_text_unwrap(text_hash)
        return Sentence(
            talk_sentence_id=None,
            name=None,
            name_hash=None,
            text=text,
            text_hash=text_hash,
        )

    def __process_didx_data(
        self, iterable: Iterable[Tuple[int, int]]
    ) -> Iterator[Audio]:
        df = self.didx_table
        existing_eids = set(df["didx_entry_eid"])

        id_to_name_sentence_map = {}
        for text_hash, voice_id in iterable:
            sentence = self.text_hash_to_sentence(text_hash)

            voice_path = self.voice_id_to_voice_path_map[voice_id]
            for postfix in ["", "_f", "_m"]:
                voice_path_variant = f"{voice_path}{postfix}"
                key = self.voice_path_to_id(voice_path_variant)
                if key in existing_eids:
                    id_to_name_sentence_map[key] = voice_path_variant, sentence
                    break

        mask = df["didx_entry_eid"].isin(id_to_name_sentence_map)
        df = df.loc[mask, ["wem_dir", "didx_entry_eid"]]

        for wem_dir, didx_entry_eid in df.itertuples(index=False):
            name, sentence = id_to_name_sentence_map[didx_entry_eid]
            path_to_wem = Path(wem_dir) / f"{didx_entry_eid:016x}.wem"
            yield Audio(name=name, sentence=sentence, wem_path=path_to_wem)

    def __process_banks_data(
        self, iterable: Iterable[Tuple[int, int]]
    ) -> Iterator[Audio]:
        id_to_name_sentence_map = {}
        for text_hash, audio_event in iterable:
            sentence = self.text_hash_to_sentence(text_hash)
            key = self.audio_event_to_id(audio_event)
            id_to_name_sentence_map[key] = audio_event, sentence

        df = self.banks_table
        mask = df["bank_entry_id"].isin(id_to_name_sentence_map)
        df = df.loc[mask, ["wem_dir", "bank_entry_id", "wem_ref_id"]]

        for wem_dir, bank_entry_id, wem_ref_id in df.itertuples(index=False):
            name, sentence = id_to_name_sentence_map[bank_entry_id]
            path_to_wem = Path(wem_dir) / f"{wem_ref_id:08x}.wem"
            yield Audio(name=name, sentence=sentence, wem_path=path_to_wem)

    @functools.cache
    def compute_voice_id_set(self, voice_path_regex: Pattern) -> pd.DataFrame:
        return {
            voice_id
            for voice_id, voice_path in self.voice_id_to_voice_path_map.items()
            if voice_path_regex.search(voice_path)
        }

    def build_talk_sentence_config(self, voice_path_regex: Pattern) -> Iterator[Audio]:
        voice_id_set = self.compute_voice_id_set(voice_path_regex)

        df = self.talk_sentence_config_table
        mask = df["voice_id"].isin(voice_id_set)
        df = df[mask]

        yield from self.__process_didx_data(
            df[["talk_sentence_text", "voice_id"]].itertuples(index=False)
        )

    def build_voice_atlas(self, avatar_id: int) -> Iterator[Audio]:
        df = self.voice_atlas_table
        mask = df["avatar_id"] == avatar_id
        df = df[mask]

        yield from self.__process_didx_data(
            df[["voice_m_hash", "audio_id"]].dropna().itertuples(index=False)
        )

        yield from self.__process_banks_data(
            df[["voice_m_hash", "audio_event"]].dropna().itertuples(index=False)
        )

    def compute_talk_sentence_config_merged_text_hash(
        self, voice_path_regex: Pattern
    ) -> List[List[int]]:
        voice_id_set = self.compute_voice_id_set(voice_path_regex)

        df = self.talk_sentence_config_table.fillna(None)
        voice_id_mask = df["voice_id"].isin(voice_id_set)
        group_id_set = set(df.loc[voice_id_mask, "group"])
        name_hash_set = set(df.loc[voice_id_mask, "textmap_talk_sentence_name"])
        mask = df["group"].isin(group_id_set)

        fields = [
            "voice_id",
            "textmap_talk_sentence_name",
            "talk_sentence_text",
            "group",
        ]

        result = []
        for _, sub_df in df.loc[mask, fields].groupby("group"):
            current_name_hash = None
            buffer = []
            for voice_id, name_hash, text_hash, _ in sub_df.itertuples(index=False):
                if name_hash == current_name_hash and voice_id in voice_id_set:
                    buffer.append(text_hash)
                    continue

                if buffer:
                    result.append(buffer)
                    buffer = []

                current_name_hash = None

                if name_hash in name_hash_set and voice_id in voice_id_set:
                    current_name_hash = name_hash
                    buffer.append(text_hash)

            if buffer:
                result.append(buffer)

        return result
