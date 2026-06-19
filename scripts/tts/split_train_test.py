import argparse
import random
import sys
from pathlib import Path

SEED = 42
TEST_RATIO = 0.10
N_REPEATS = 3


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle a JSONL with a fixed seed and split into 90% train / 10% test shards."
    )
    parser.add_argument("--jsonl", required=True)
    args = parser.parse_args()

    src = Path(args.jsonl).resolve()
    assert src.is_file(), f"Source JSONL not found: {src}"

    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"[info] {src}: {len(lines)} non-empty entries", file=sys.stderr)

    n_test = max(1, round(len(lines) * TEST_RATIO)) if lines else 0

    out_root = src.parent
    random.seed(SEED)

    for shard in range(N_REPEATS):
        random.shuffle(lines)
        test_lines = lines[:n_test]
        train_lines = lines[n_test:]

        train_out = out_root / "train" / str(shard) / src.name
        test_out = out_root / "test" / str(shard) / src.name
        train_out.parent.mkdir(parents=True, exist_ok=True)
        test_out.parent.mkdir(parents=True, exist_ok=True)

        train_out.write_text("".join(ln + "\n" for ln in train_lines), encoding="utf-8")
        test_out.write_text("".join(ln + "\n" for ln in test_lines), encoding="utf-8")

        print(
            f"[done] shard {shard}: train {len(train_lines)} -> {train_out} | "
            f"test {len(test_lines)} -> {test_out}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
