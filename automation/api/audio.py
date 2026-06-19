import os
import subprocess
from pathlib import Path
from typing import Mapping, Any

from pydantic import BaseModel, ConfigDict

from automation.api.out_trait import OutTrait
from automation.api.sentence import Sentence

VGMSTREAM = os.environ.get("VGMSTREAM", "vgmstream-cli")


class Audio(BaseModel, OutTrait):
    model_config = ConfigDict(frozen=True)

    name: str
    sentence: Sentence
    wem_path: Path

    @property
    def num_tokens(self) -> int:
        return self.sentence.num_tokens

    @property
    def dot_wav(self) -> str:
        return f"{self.name}.wav"

    def wem_to_wav_by_vgmstream(self, output_dir: Path) -> None:
        assert VGMSTREAM is not None
        assert output_dir.exists()
        wav_path = output_dir / self.dot_wav
        result = subprocess.run(
            [VGMSTREAM, "-o", str(wav_path), str(self.wem_path)],
            capture_output=True,
        )
        assert result.returncode == 0

    def to_jsonl(self, **_: Any) -> Mapping[str, Any]:
        return {
            "audio": self.dot_wav,
            "text": self.sentence.pretty_string,
        }
