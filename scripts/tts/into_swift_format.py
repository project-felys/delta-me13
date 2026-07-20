import argparse
import json
import shutil
import sys
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm

TARGET_SR = 24000


def find_longest_audio(wav_dir, names):
    best_name, best_dur = None, -1.0
    for name in names:
        info = sf.info(str(Path(wav_dir) / name))
        dur = info.frames / info.samplerate
        if dur > best_dur:
            best_name, best_dur = name, dur
    return best_name, best_dur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    src_jsonl = Path(args.jsonl).resolve()
    src_wav_dir = src_jsonl.parent / src_jsonl.stem
    assert src_jsonl.is_file(), f"Source JSONL not found: {src_jsonl}"
    assert src_wav_dir.is_dir(), f"Source wav directory not found: {src_wav_dir}"

    out_dir = Path(args.output_dir).resolve()
    out_wav_dir = out_dir / src_jsonl.stem
    out_jsonl = out_dir / src_jsonl.name
    out_ref = out_dir / "ref.wav"
    out_wav_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        json.loads(line) for line in open(src_jsonl, encoding="utf-8") if line.strip()
    ]
    print(f"[info] {src_jsonl}: {len(entries)} entries", file=sys.stderr)

    longest_name, longest_dur = find_longest_audio(
        src_wav_dir, [e["audio"] for e in entries]
    )
    print(
        f"[info] longest clip -> {longest_name} ({longest_dur:.2f}s), used as ref_audio",
        file=sys.stderr,
    )

    for e in tqdm(entries, desc="resample", unit="wav"):
        y, _ = librosa.load(str(src_wav_dir / e["audio"]), sr=TARGET_SR, mono=True)
        sf.write(str(out_wav_dir / e["audio"]), y, TARGET_SR, subtype="PCM_16")

    shutil.copyfile(out_wav_dir / longest_name, out_ref)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for e in entries:
            rec = {
                "messages": [{"role": "assistant", "content": e["text"].strip()}],
                "audios": [str((out_wav_dir / e["audio"]).resolve())],
                "ref_audios": [str(out_ref.resolve())],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[done] wrote {len(entries)} lines -> {out_jsonl}")
    print(f"ref_audio -> {out_ref} (from {longest_name}, {longest_dur:.2f}s)")


if __name__ == "__main__":
    main()
