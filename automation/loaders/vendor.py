import functools
import json
import os
from pathlib import Path
from typing import Mapping

import pandas as pd


class VendorLoader:
    def __init__(self, vendor_dir: Path):
        self.__vendor_dir = vendor_dir

    @functools.cached_property
    def miyoushe(self) -> Mapping[str, str]:
        path_to_miyoushe = self.__vendor_dir / "miyoushe"
        name_to_description = {}
        for file_name in os.listdir(path_to_miyoushe):
            file_path = path_to_miyoushe / file_name
            if not file_path.is_file():
                continue
            with open(file_path) as file:
                name_to_description[file_name] = file.read()
        return name_to_description

    @functools.cached_property
    def coig(self) -> pd.DataFrame:
        with open(self.__vendor_dir / "coig" / "leetcode_instructions.jsonl") as f:
            data = (json.loads(line) for line in f.readlines())
            return pd.DataFrame(data=data)
