from pathlib import Path
import re
from typing import Iterable, Iterator, Mapping, Pattern, Tuple

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


class AudioFactory(UnpackedAudioLanguageLoader, TurnBasedGameDataLoader):
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
        self.__unpacked_audio_language = language

    @property
    def cyrene(self) -> Mapping[str, Iterator[Audio]]:
        voice_path_regex = re.compile(
            r"(?:"
            r"(?:chapter4|side4)_[^_]+"
            r"|vo_ambient_w4_\w+_\w+"
            r")"
            r"_(?:cyrene|cyrenejiyi|cyrenely|wangxi|zuozhe)_\d+"
        )

        return {
            "talk_sentence_config": self.build_talk_sentence_config(voice_path_regex),
            "voice_atlas": self.build_voice_atlas(1415),
        }

    def __voice_path_to_id(self, voice_path: str) -> int:
        wem_path = f"{self.__unpacked_audio_language}/voice/{voice_path}.wem"
        return fnv1_64(wem_path.lower())

    def __audio_event_to_id(self, audio_event: str) -> int:
        event_id = fnv1_32(audio_event.lower())
        return self.event_id_to_first_bank_map[event_id]

    def __process_didx_data(
        self, iterable: Iterable[Tuple[int, int]]
    ) -> Iterator[Audio]:
        df = self.didx_table
        existing_eids = set(df["didx_entry_eid"])

        id_to_name_sentence_map = {}
        for text_hash, voice_id in iterable:
            text = self.get_text_unwrap(text_hash)
            sentence = Sentence.plain_text(text)

            voice_path = self.voice_id_to_voice_path_map[voice_id]
            for postfix in ["", "_f", "_m"]:
                voice_path_variant = f"{voice_path}{postfix}"
                key = self.__voice_path_to_id(voice_path_variant)
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
            text = self.get_text_unwrap(text_hash)
            sentence = Sentence.plain_text(text)

            key = self.__audio_event_to_id(audio_event)
            id_to_name_sentence_map[key] = audio_event, sentence

        df = self.banks_table
        mask = df["bank_entry_id"].isin(id_to_name_sentence_map)
        df = df.loc[mask, ["wem_dir", "bank_entry_id", "wem_ref_id"]]

        for wem_dir, bank_entry_id, wem_ref_id in df.itertuples(index=False):
            name, sentence = id_to_name_sentence_map[bank_entry_id]
            path_to_wem = Path(wem_dir) / f"{wem_ref_id:08x}.wem"
            yield Audio(name=name, sentence=sentence, wem_path=path_to_wem)

    def build_talk_sentence_config(self, voice_path_regex: Pattern) -> Iterator[Audio]:
        voice_id_set = {
            voice_id
            for voice_id, voice_path in self.voice_id_to_voice_path_map.items()
            if voice_path_regex.search(voice_path)
        }
        df = self.talk_sentence_config_table
        mask = df["voice_id"].isin(voice_id_set)
        df = df.loc[mask, ["talk_sentence_text", "voice_id"]]

        yield from self.__process_didx_data(df.itertuples(index=False))

    def build_voice_atlas(self, avatar_id: int) -> Iterator[Audio]:
        df = self.voice_atlas_table
        mask = df["avatar_id"] == avatar_id
        df = df[mask]

        didx_df = df[["voice_m_hash", "audio_id"]].dropna()
        banks_df = df[["voice_m_hash", "audio_event"]].dropna()

        yield from self.__process_didx_data(didx_df.itertuples(index=False))
        yield from self.__process_banks_data(banks_df.itertuples(index=False))
