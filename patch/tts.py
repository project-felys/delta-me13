import json
from pathlib import Path
from typing import Mapping


def load_patch_file(input_path: Path) -> Mapping[str, str]:
    audio_to_patch_map = {}
    with open(input_path) as file:
        for line in file:
            patch = json.loads(line)
            audio_to_patch_map[patch["audio"]] = patch["patch"]
    return audio_to_patch_map


def load_patch(name: str) -> Mapping[str, str]:
    input_path = Path(__file__).parent / "tts" / f"{name}.jsonl"
    return load_patch_file(input_path) if input_path.exists() else {}
