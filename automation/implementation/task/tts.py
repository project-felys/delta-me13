import itertools
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, List

from automation.api.audio import Audio
from automation.api.sentence import Sentence
from automation.factories.tts import TtsFactory
from automation.implementation.auto_format import get_patch_auto_format
from automation.implementation.match_sub import get_slow_path_only_match_sub
from automation.implementation.task.out_trait import emit


def __convert_audio(
    iterable: Iterator[Audio], output_dir: Path, num_threads: int
) -> Iterator[Audio]:
    def worker(audio: Audio) -> Audio:
        audio.wem_to_wav_by_vgmstream(output_dir)
        return audio

    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        for each in ex.map(worker, iterable):
            yield each


def __ensure_dir_exist(output_dir: Path) -> None:
    os.makedirs(output_dir, exist_ok=True)


def cyrene(
    output_dir: Path,
    unpacked_audio_dir: Path,
    turn_based_game_data_dir: Path,
    language: str,
    num_threads: int,
) -> List[int]:
    Sentence.global_config(
        auto_format=get_patch_auto_format(language),
        token_counter=None,
        match_sub=get_slow_path_only_match_sub(),
    )

    unpacked_audio_language_dir = unpacked_audio_dir / language
    audio_dir = output_dir / language
    __ensure_dir_exist(unpacked_audio_language_dir)
    __ensure_dir_exist(audio_dir)

    voice_path_regex = re.compile(
        r"(?:"
        r"(?:chapter4|side4)_[^_]+"
        r"|vo_ambient_w4_\w+_\w+"
        r")"
        r"_(?:cyrene|cyrenejiyi|cyrenely|wangxi|zuozhe)_\d+"
    )

    factory = TtsFactory(unpacked_audio_language_dir, turn_based_game_data_dir)
    merge_data = factory.compute_talk_sentence_config_merged_text_hash(voice_path_regex)
    with open(output_dir / f"{language}.json", "w") as file:
        json.dump(merge_data, file, indent=2)

    iterable = itertools.chain(
        factory.build_talk_sentence_config(voice_path_regex),
        factory.build_voice_atlas(1415),
    )
    iterable = __convert_audio(iterable, audio_dir, num_threads)
    return emit(iterable, output_dir / f"{language}.jsonl", f"{language:>12}")
