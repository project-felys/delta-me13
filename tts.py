import argparse
import json
from pathlib import Path
from typing import Any, List, Mapping, Callable

import numpy as np
import soundfile as sf
import torch

from clearvoice import ClearVoice
import tqdm

from patch.tts import load_patch
from automation.sdk.token_counter import get_qwen3_token_counter

MODEL_SAMPLE_RATE = 48000
MODEL_NAME = "MossFormer2_SE_48K"


def load_model() -> ClearVoice:
    return ClearVoice(task="speech_enhancement", model_names=[MODEL_NAME])


def audio_long_enough(input_path: Path, min_seconds: float) -> bool:
    info = sf.info(str(input_path))
    return min_seconds <= info.frames / info.samplerate


def text_long_enough(text: str, token_counter: Callable[[str], int]) -> bool:
    return 4 <= token_counter(text)


def clear_voice(input_path: Path, output_dir: Path, model: ClearVoice):
    audio, sr = sf.read(str(input_path))
    assert audio.ndim == 1
    assert sr == MODEL_SAMPLE_RATE
    audio = audio.astype(np.float32)[None, :]
    with torch.no_grad():
        data = model(audio).squeeze()
    sf.write(str(output_dir / input_path.name), data, MODEL_SAMPLE_RATE)


def load_audio_jsonl(jsonl_path: Path) -> List[Mapping[str, Any]]:
    with open(jsonl_path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-seconds", type=float, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    jsonl_path: Path = args.jsonl
    output_dir: Path = args.output_dir

    name = jsonl_path.stem

    input_wav_dir = jsonl_path.parent / name
    output_wav_dir = output_dir / name
    output_wav_dir.mkdir(parents=True, exist_ok=True)

    token_counter = get_qwen3_token_counter()
    audio_jsonl = load_audio_jsonl(jsonl_path)
    patch_mapping = load_patch(name)
    model = load_model()

    with open(output_dir / f"{name}.jsonl", "w") as file:
        for entry in tqdm.tqdm(audio_jsonl):
            audio = entry["audio"]
            text = patch_mapping.get(audio, entry["text"])

            path_to_wav = input_wav_dir / audio

            if not audio_long_enough(path_to_wav, args.min_seconds):
                continue

            if not text_long_enough(text, token_counter):
                continue

            output_wav = output_wav_dir / audio
            if args.force or not output_wav.exists():
                clear_voice(path_to_wav, output_wav_dir, model)

            output_json = {"audio": audio, "text": text}
            line = json.dumps(output_json, ensure_ascii=False)
            print(line, file=file)
