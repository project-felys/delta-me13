import functools
import os
from pathlib import Path
from typing import Mapping

import pandas as pd

from pck import PckHeader


class UnpackedAudioLanguageLoader:
    def __init__(self, unpacked_audio_dir: Path):
        self.__unpacked_audio_language_dir = unpacked_audio_dir

    @functools.cached_property
    def __pck_headers(self) -> Mapping[str, PckHeader]:
        wildcard_json = (
            self.__unpacked_audio_language_dir / file
            for file in os.listdir(self.__unpacked_audio_language_dir)
            if file.endswith(".json")
        )
        return {
            str(path).rstrip(".json"): PckHeader.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            for path in wildcard_json
        }

    @functools.cached_property
    def didx_table(self) -> pd.DataFrame:
        df = pd.DataFrame(
            data=(
                (wem_dir, didx_entry.eid)
                for wem_dir, pck_header in self.__pck_headers.items()
                for didx_entry in pck_header.didx
            ),
            columns=["wem_dir", "didx_entry_eid"],
        )
        df.values.flags.writeable = False
        return df

    @functools.cached_property
    def banks_table(self) -> pd.DataFrame:
        df = pd.DataFrame(
            data=(
                (wem_dir, bank_entry.id, wem_ref.id)
                for wem_dir, pck_header in self.__pck_headers.items()
                for bank_entry in pck_header.banks
                for wem_ref in bank_entry.wems
            ),
            columns=["wem_dir", "bank_entry_id", "wem_ref_id"],
        )
        df.values.flags.writeable = False
        return df
