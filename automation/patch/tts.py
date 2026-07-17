import json
from pathlib import Path
from typing import Mapping


def load_patch_file(input_path: Path) -> Mapping[str, str]:
    hash_to_text_map = {}
    with open(input_path) as file:
        for line in file:
            patch = json.loads(line)
            hash_to_text_map[patch["hash"]] = patch["patch"]
    return hash_to_text_map


def load_patch(name: str) -> Mapping[str, str]:
    input_path = Path(__file__).parent / "tts" / f"{name}.jsonl"
    return load_patch_file(input_path) if input_path.exists() else {}
