import argparse
import struct
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field
import tqdm


class Folder(BaseModel):
    id: int
    name: str


class WemRef(BaseModel):
    id: int
    size: int


class BankEntry(BaseModel):
    id: int
    lang: int
    size: int
    offset: int
    flags: int
    wems: List[WemRef] = Field(default_factory=list)


class StmEntry(BaseModel):
    id: int
    lang: int
    size: int
    offset: int
    flags: int


class DidxEntry(BaseModel):
    eid: int
    size: int
    offset: int
    folder: int


class PckHeader(BaseModel):
    header_size: int
    data_start: int
    folders: List[Folder]
    bank_tab_size: int
    stm_tab_size: int
    didx_tab_size: int
    banks: List[BankEntry]
    stms: List[StmEntry]
    didx: List[DidxEntry]


class SourceMeta(BaseModel):
    filename: str
    path: str
    size: int


class PckExtractor:
    def __init__(self, pck_path: Path):
        assert pck_path.is_file()
        self.__path = pck_path
        self.__data = pck_path.read_bytes()
        assert self.__data[:4] == b"AKPK"

    def __u32(self, offset: int) -> int:
        return struct.unpack_from("<I", self.__data, offset)[0]

    def __parse_folders(self) -> List[Folder]:
        data = self.__data
        base = 0x1C
        offset = base + 4
        folders: List[Folder] = []
        for _ in range(self.__u32(base)):
            str_offset, fid = struct.unpack_from("<II", data, offset)
            offset += 8
            start = base + str_offset
            end = start
            while end + 2 <= len(data) and data[end : end + 2] != b"\x00\x00":
                end += 2
            folders.append(
                Folder(id=fid, name=data[start:end].decode("utf-16le", "replace"))
            )
        return folders

    def __parse_header(self) -> PckHeader:
        data = self.__data
        header_size = self.__u32(0x04)
        bank_tab_size = self.__u32(0x10)
        stm_tab_size = self.__u32(0x14)
        didx_tab_size = self.__u32(0x18)
        folders = self.__parse_folders()
        data_start = 8 + header_size

        offset = data_start - bank_tab_size - stm_tab_size - didx_tab_size

        bank_count = self.__u32(offset)
        offset += 4
        banks: List[BankEntry] = []
        for _ in range(bank_count):
            bid, lang, size, boff, flags = struct.unpack_from("<IIIII", data, offset)
            entry = BankEntry(id=bid, lang=lang, size=size, offset=boff, flags=flags)
            banks.append(entry)
            offset += 20

        stm_count = self.__u32(offset)
        offset += 4
        stms: List[StmEntry] = []
        for _ in range(stm_count):
            sid, lang, size, soff, flags = struct.unpack_from("<IIIII", data, offset)
            entry = StmEntry(id=sid, lang=lang, size=size, offset=soff, flags=flags)
            stms.append(entry)
            offset += 20

        didx_count = self.__u32(offset)
        offset += 4
        didx: List[DidxEntry] = []
        for _ in range(didx_count):
            eid, mult, size, doff, folder = struct.unpack_from("<QIIII", data, offset)
            entry = DidxEntry(eid=eid, size=size, offset=doff * mult, folder=folder)
            didx.append(entry)
            offset += 24

        return PckHeader(
            header_size=header_size,
            data_start=data_start,
            folders=folders,
            bank_tab_size=bank_tab_size,
            stm_tab_size=stm_tab_size,
            didx_tab_size=didx_tab_size,
            banks=banks,
            stms=stms,
            didx=didx,
        )

    @staticmethod
    def __parse_soundbank(blob: bytes) -> List[Tuple[int, bytes]]:
        chunks: Dict[bytes, Tuple[int, int]] = {}
        offset = 0
        while offset + 8 <= len(blob):
            tag = blob[offset : offset + 4]
            if not all(0x20 <= b < 0x7F for b in tag):
                break
            size = struct.unpack_from("<I", blob, offset + 4)[0]
            chunks[tag] = (offset + 8, size)
            offset += 8 + size

        data = chunks.get(b"DATA")
        didx = chunks.get(b"DIDX")
        if data is None or didx is None:
            return []

        data_off, _ = data
        didx_off, didx_size = didx
        wems: List[Tuple[int, bytes]] = []
        for i in range(didx_size // 12):
            wid, woff, wsz = struct.unpack_from("<III", blob, didx_off + i * 12)
            wems.append((wid, blob[data_off + woff : data_off + woff + wsz]))
        return wems

    def __extract_bank(self, entry: BankEntry, wem_dir: Path) -> None:
        blob = self.__data[entry.offset : entry.offset + entry.size]
        for wid, wem in self.__parse_soundbank(blob):
            (wem_dir / f"{wid:08x}.wem").write_bytes(wem)
            entry.wems.append(WemRef(id=wid, size=len(wem)))

    def __extract_stm(self, entry: StmEntry, wem_dir: Path) -> None:
        blob = self.__data[entry.offset : entry.offset + entry.size]
        (wem_dir / f"{entry.id:08x}.wem").write_bytes(blob)

    def __extract_didx(self, entry: DidxEntry, wem_dir: Path) -> None:
        blob = self.__data[entry.offset : entry.offset + entry.size]
        (wem_dir / f"{entry.eid:016x}.wem").write_bytes(blob)

    def extract(self, output_dir: Path) -> None:
        header = self.__parse_header()
        wem_dir = output_dir / self.__path.stem
        wem_dir.mkdir(parents=True, exist_ok=True)

        for entry in header.banks:
            self.__extract_bank(entry, wem_dir)
        for entry in header.stms:
            self.__extract_stm(entry, wem_dir)
        for entry in header.didx:
            self.__extract_didx(entry, wem_dir)

        json_path = output_dir / f"{self.__path.stem}.json"
        json_output = header.model_dump_json(indent=2, ensure_ascii=False)
        json_path.write_text(json_output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("audio"))
    args = parser.parse_args()

    all_sub_dirs = [entry for entry in args.input_dir.iterdir() if entry.is_dir()]
    name_width = max((len(d.name) for d in all_sub_dirs), default=0)

    for sub_dir in all_sub_dirs:
        wildcard_dot_pck = [
            entry for entry in sub_dir.iterdir() if entry.suffix == ".pck"
        ]

        output_dir = args.output_dir / sub_dir.name
        desc = f"> {sub_dir.name.rjust(name_width)}"
        for dot_pck in tqdm.tqdm(wildcard_dot_pck, desc=desc):
            PckExtractor(dot_pck).extract(output_dir)


if __name__ == "__main__":
    main()
