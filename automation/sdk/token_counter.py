from pathlib import Path
from transformers import AutoTokenizer
from typing import Callable


def get_qwen3_token_counter() -> Callable[[str], int]:
    tokenizer_path = Path(__file__).parent / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, local_files_only=True
    )
    return lambda x: len(tokenizer.encode(x))
