import functools
import itertools
import json
import re
from pathlib import Path
from typing import Any, List, Mapping, Tuple

import pandas as pd

from automation.api.sentence import Sentence


def get_into_hash(config: Mapping[str, int], key: str):
    target_dict = config.get(key)
    hash_value = pd.NA
    if target_dict:
        hash_value = target_dict["Hash"]
    return hash_value


class TurnBasedGameLoader:
    __text_join_pattern: re.Pattern[str] = re.compile(r"\{TEXTJOIN#(\d+)\}")

    def __init__(self, root_dir: Path, language: str):
        self.__root_dir = root_dir
        self.__language = language
        self.__visited_textmap_keys_set = set()

    @property
    def utilization(self):
        return len(self.__visited_textmap_keys_set) / len(self.__text_map)

    @functools.cached_property
    def __talk_sentence_config(self) -> List[Mapping[str, Any]]:
        with open(self.__root_dir / "ExcelOutput/TalkSentenceConfig.json") as f:
            return json.load(f)

    @functools.cached_property
    def talk_sentence_config_table(self) -> pd.DataFrame:
        df = pd.DataFrame(
            data=(
                (
                    entry.get("TalkSentenceID", pd.NA),
                    get_into_hash(entry, "TextmapTalkSentenceName"),
                    get_into_hash(entry, "TalkSentenceText"),
                )
                for entry in self.__talk_sentence_config
            ),
            columns=[
                "talk_sentence_id",
                "textmap_talk_sentence_name",
                "talk_sentence_text",
            ],
        )

        mask = df["talk_sentence_text"].isin(self.__text_map)
        df = df[mask]
        df["group"] = df["talk_sentence_id"] // 1000
        df = df.fillna(None).sort_values("talk_sentence_id")

        df.values.flags.writeable = False
        return df

    def sentence_factory(
        self, talk_sentence_id: int, name_hash: int, text_hash: int
    ) -> Sentence:
        return Sentence(
            talk_sentence_id=talk_sentence_id,
            name=name_hash and self.get_joined_text(name_hash, None) or "{NICKNAME}",
            name_hash=name_hash,
            text=self.get_joined_text_unwrap(text_hash),
            text_hash=text_hash,
        )

    @functools.cached_property
    def __text_map(self) -> Mapping[int, str]:
        text_map_path = self.__root_dir / f"TextMap/TextMap{self.__language}.json"

        if text_map_path.exists():
            with open(text_map_path) as f:
                text_map = json.load(f)
                return {int(k): v for k, v in text_map.items()}

        text_map = {}
        i = 0
        while True:
            partial_path = (
                self.__root_dir / f"TextMap/TextMap{self.__language}_{i}.json"
            )
            if not partial_path.exists():
                break

            with open(partial_path) as f:
                partial_text_map = json.load(f)

            text_map |= {int(k): v for k, v in partial_text_map.items()}
            i += 1

        return text_map

    @functools.cached_property
    def __text_join_config(self) -> List[Mapping[str, Any]]:
        with open(self.__root_dir / "ExcelOutput/TextJoinConfig.json") as f:
            return json.load(f)

    @functools.cached_property
    def __text_join_item(self) -> List[Mapping[str, Any]]:
        with open(self.__root_dir / "ExcelOutput/TextJoinItem.json") as f:
            return json.load(f)

    @functools.cached_property
    def __text_join_id_to_text(self) -> Mapping[int, str]:
        item_id_to_text = {
            item["TextJoinItemID"]: self.__text_map[item["TextJoinText"]["Hash"]]
            for item in self.__text_join_item
            if "TextJoinText" in item
        }

        text_join_id_to_text = {}
        for item in self.__text_join_config:
            text = item_id_to_text.get(item["DefaultItem"])
            if text is None:
                text_join_item_list = sorted(item["TextJoinItemList"])
                for tid in text_join_item_list:
                    text = item_id_to_text.get(tid)
                    if text is not None:
                        break
            if text is not None:
                text_join_id_to_text[item["TextJoinID"]] = text

        return text_join_id_to_text

    def __replace_text_join(self, match: re.Match[str]) -> str:
        key = int(match.group(1))
        return self.__text_join_id_to_text.get(key, match.group(0))

    def __join_text(self, s: str) -> str:
        s = self.__text_join_pattern.sub(self.__replace_text_join, s)
        s = s.replace("\\n", "\n")
        return s

    def get_joined_text(self, key: int, default: Any):
        try:
            return self.get_joined_text_unwrap(key)
        except:
            return default

    def get_joined_text_unwrap(self, key: int):
        raw_text = self.__text_map[key]
        self.__visited_textmap_keys_set.add(key)
        return self.__join_text(raw_text)

    @functools.cached_property
    def __book_series_config(self) -> List[Mapping[str, Any]]:
        with open(self.__root_dir / "ExcelOutput/BookSeriesConfig.json") as f:
            return json.load(f)

    @functools.cached_property
    def __local_book_config(self) -> List[Mapping[str, Any]]:
        with open(self.__root_dir / "ExcelOutput/LocalbookConfig.json") as f:
            return json.load(f)

    @functools.cached_property
    def book_series_config_table(self) -> pd.DataFrame:
        df = pd.DataFrame(
            data=(
                (
                    entry["BookSeriesID"],
                    entry["BookSeries"]["Hash"],
                    get_into_hash(entry, "BookSeriesComments"),
                    entry["BookSeriesWorld"],
                )
                for entry in self.__book_series_config
            ),
            columns=[
                "book_series_id",
                "book_series_hash",
                "book_series_comments_hash",
                "book_series_world",
            ],
        )
        df.values.flags.writeable = False
        return df

    @functools.cached_property
    def local_book_config_table(self):
        df = pd.DataFrame(
            data=(
                (
                    entry["BookSeriesID"],
                    entry["BookInsideName"]["Hash"],
                    entry["BookContent"]["Hash"],
                )
                for entry in self.__local_book_config
            ),
            columns=[
                "book_series_id",
                "book_inside_name_hash",
                "book_content_hash",
            ],
        )
        df.values.flags.writeable = False
        return df

    @functools.cached_property
    def __story_atlas(self) -> List[Mapping[str, Any]]:
        with open(self.__root_dir / "ExcelOutput/StoryAtlas.json") as f:
            return json.load(f)

    @functools.cached_property
    def story_atlas_table(self) -> pd.DataFrame:
        df = pd.DataFrame(
            (
                (
                    entry["AvatarID"],
                    entry["StoryID"],
                    entry["Story"]["Hash"],
                    entry.get("ReplaceID", pd.NA),
                )
                for entry in self.__story_atlas
            ),
            columns=["avatar_id", "story_id", "story_hash", "replace_id"],
        )
        df.values.flags.writeable = False
        return df

    @functools.cached_property
    def __avatar_config(self) -> List[Mapping[str, Any]]:
        with open(self.__root_dir / "ExcelOutput/AvatarConfig.json") as f:
            return json.load(f)

    @functools.cached_property
    def __avatar_config_ld(self) -> List[Mapping[str, Any]]:
        with open(self.__root_dir / "ExcelOutput/AvatarConfigLD.json") as f:
            return json.load(f)

    @functools.cached_property
    def avatar_id_to_name_map(self) -> Mapping[int, str]:
        return {
            entry["AvatarID"]: self.get_joined_text_unwrap(entry["AvatarName"]["Hash"])
            for entry in itertools.chain(self.__avatar_config, self.__avatar_config_ld)
        }

    @functools.cached_property
    def __tarot_mails(self) -> List[Mapping[str, Any]]:
        with open(self.__root_dir / "ExcelOutput/TarotMails.json") as f:
            return json.load(f)

    @functools.cached_property
    def mail_sentence_id_to_sentence_map(self) -> Mapping[int, str]:
        return {
            entry["ID"]: self.get_joined_text_unwrap(entry["Sentence"]["Hash"])
            for entry in self.__tarot_mails
        }

    @functools.cached_property
    def __tarot_mailbox(self) -> List[Mapping[str, Any]]:
        with open(self.__root_dir / "ExcelOutput/TarotMailbox.json") as f:
            return json.load(f)

    @functools.cached_property
    def tarot_mailbox_table(self) -> pd.DataFrame:
        df = pd.DataFrame(
            (
                (
                    entry["Title"]["Hash"],
                    entry["From"]["Hash"],
                    entry["To"]["Hash"],
                    entry["MailSentenceIDList"],
                )
                for entry in self.__tarot_mailbox
            ),
            columns=["title_hash", "from_hash", "to_hash", "mail_sentence_id_list"],
        )
        df.values.flags.writeable = False
        return df

    @functools.cached_property
    def __tarot_book_sentence(self) -> List[Mapping[str, Any]]:
        with open(self.__root_dir / "ExcelOutput/TarotBookSentence.json") as f:
            return json.load(f)

    @functools.cached_property
    def tarot_book_sentence_table(self) -> pd.DataFrame:
        df = pd.DataFrame(
            data=(
                (
                    entry["ID"],
                    entry["Sentence"]["Hash"],
                )
                for entry in self.__tarot_book_sentence
            ),
            columns=[
                "id",
                "sentence_id",
            ],
        )

        df = df.sort_values("id")
        df.values.flags.writeable = False
        return df

    @functools.cached_property
    def __none_atlas(self) -> List[Mapping[str, Any]]:
        with open(self.__root_dir / "ExcelOutput/NounAtlas.json") as f:
            return json.load(f)

    @functools.cached_property
    def none_atlas_table(self) -> pd.DataFrame:
        df = pd.DataFrame(
            data=(
                (
                    config["NounTitle"]["Hash"],
                    config["NounDesc"]["Hash"],
                )
                for config in self.__none_atlas
            ),
            columns=[
                "noun_title_hash",
                "noun_desc_hash",
            ],
        )
        df.values.flags.writeable = False
        return df

    @functools.cached_property
    def __chronicle_conclusion(self) -> List[Mapping[str, Any]]:
        with open(self.__root_dir / "ExcelOutput/ChronicleConclusion.json") as f:
            return json.load(f)

    @functools.cached_property
    def chronicle_conclusion_table(self) -> pd.DataFrame:
        df = pd.DataFrame(
            data=(
                (
                    entry["MissionID"],
                    entry["MissionConclusion"]["Hash"],
                )
                for entry in self.__chronicle_conclusion
            ),
            columns=[
                "mission_id",
                "mission_conclusion_hash",
            ],
        )
        df = df.sort_values("mission_id")
        df.values.flags.writeable = False
        return df

    @functools.cached_property
    def __sub_mission(self) -> List[Mapping[str, Any]]:
        with open(self.__root_dir / "ExcelOutput/SubMission.json") as f:
            return json.load(f)

    @functools.cached_property
    def sub_mission_table(self) -> pd.DataFrame:
        df = pd.DataFrame(
            data=(
                (
                    entry["SubMissionID"],
                    get_into_hash(entry, "TargetText"),
                    get_into_hash(entry, "DescrptionText"),
                )
                for entry in self.__sub_mission
            ),
            columns=[
                "sub_mission_id",
                "target_text_hash",
                "descrption_text_hash",
            ],
        )
        df = df.dropna(subset=["target_text_hash"]).sort_values("sub_mission_id")
        df.values.flags.writeable = False
        return df

    @functools.cached_property
    def __main_mission(self) -> List[Mapping[str, Any]]:
        with open(self.__root_dir / "ExcelOutput/MainMission.json") as f:
            return json.load(f)

    @functools.cached_property
    def main_mission_id_to_name_map(self) -> Mapping[str, str]:
        return {
            entry["MainMissionID"]: self.get_joined_text(entry["Name"]["Hash"], None)
            for entry in self.__main_mission
            if "Name" in entry
        }

    @functools.cached_property
    def __tarot_wiki_changeinfo(self) -> List[Mapping[str, Any]]:
        with open(self.__root_dir / "ExcelOutput/TarotWikiChangeinfo.json") as f:
            return json.load(f)

    @functools.cached_property
    def __tarot_wiki_changeinfo_table(self) -> pd.DataFrame:
        df = pd.DataFrame(
            data=(
                (
                    entry["ChangeID"],
                    entry["NewTitle"]["Hash"],
                    entry["NewDetails"]["Hash"],
                )
                for entry in self.__tarot_wiki_changeinfo
            ),
            columns=[
                "change_id",
                "new_title_hash",
                "new_details_hash",
            ],
        )
        df = df.set_index("change_id")
        df.values.flags.writeable = False
        return df

    def __select_title_details(
        self, title_hash: int, details_hash: int, change_id: List[int]
    ) -> Tuple[int, int]:
        if change_id:
            latest_change_id = max(change_id)
            return self.__tarot_wiki_changeinfo_table.loc[latest_change_id]
        else:
            return title_hash, details_hash

    @functools.cached_property
    def __tarot_wiki_data(self) -> List[Mapping[str, Any]]:
        with open(self.__root_dir / "ExcelOutput/TarotWikiData.json") as f:
            return json.load(f)

    @functools.cached_property
    def tarot_wiki_data_table(self) -> pd.DataFrame:
        df = pd.DataFrame(
            data=(
                (
                    entry["ID"],
                    *self.__select_title_details(
                        entry["Title"]["Hash"],
                        entry["Details"]["Hash"],
                        entry["ChangeID"],
                    ),
                    sorted(entry["SubdataList"]),
                )
                for entry in self.__tarot_wiki_data
            ),
            columns=[
                "id",
                "title_hash",
                "details_hash",
                "subdata_list",
            ],
        )
        df = df.sort_values("id")
        df.values.flags.writeable = False
        return df

    @functools.cached_property
    def __tarot_wiki_subdata(self) -> List[Mapping[str, Any]]:
        with open(self.__root_dir / "ExcelOutput/TarotWikiSubdata.json") as f:
            return json.load(f)

    @functools.cached_property
    def tarot_wiki_subdata_table(self) -> pd.DataFrame:
        df = pd.DataFrame(
            data=(
                (
                    entry["ID"],
                    *self.__select_title_details(
                        entry["Title"]["Hash"],
                        entry["Details"]["Hash"],
                        entry["ChangeID"],
                    ),
                )
                for entry in self.__tarot_wiki_subdata
            ),
            columns=[
                "id",
                "title_hash",
                "details_hash",
            ],
        )
        df = df.set_index("id")
        df.values.flags.writeable = False
        return df

    @functools.cached_property
    def __voice_atlas(self) -> List[Mapping[str, Any]]:
        with open(self.__root_dir / "ExcelOutput/VoiceAtlas.json") as f:
            return json.load(f)

    @functools.cached_property
    def voice_atlas_table(self) -> pd.DataFrame:
        df = pd.DataFrame(
            data=(
                (
                    entry["AvatarID"],
                    entry["VoiceTitle"]["Hash"],
                    entry["Voice_M"]["Hash"],
                    entry["SortID"],
                )
                for entry in self.__voice_atlas
            ),
            columns=["avatar_id", "voice_title_hash", "voice_m_hash", "sort_id"],
        )
        df.values.flags.writeable = False
        return df
