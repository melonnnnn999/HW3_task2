#!/usr/bin/env python3
"""Extract selected files from a remote ZIP/ZIP64 archive with HTTP ranges."""

from __future__ import annotations

import argparse
import pathlib
import struct
import urllib.request
import zlib
from dataclasses import dataclass


EOCD_SIG = b"PK\x05\x06"
ZIP64_LOCATOR_SIG = b"PK\x06\x07"
ZIP64_EOCD_SIG = b"PK\x06\x06"
CENTRAL_DIR_SIG = b"PK\x01\x02"
LOCAL_FILE_SIG = b"PK\x03\x04"
ZIP64_EXTRA_ID = 0x0001


@dataclass
class ZipMember:
    name: str
    compression_method: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


def fetch_range(url: str, start: int, end: int, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def content_length(url: str, timeout: int) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.headers["Content-Length"])


def parse_zip64_directory(url: str, timeout: int) -> tuple[int, int]:
    size = content_length(url, timeout)
    tail_start = max(0, size - 100_000)
    tail = fetch_range(url, tail_start, size - 1, timeout)

    eocd_pos = tail.rfind(EOCD_SIG)
    if eocd_pos < 0:
        raise RuntimeError("Could not find ZIP EOCD record.")

    locator_pos = tail.rfind(ZIP64_LOCATOR_SIG, 0, eocd_pos)
    if locator_pos < 0:
        fields = struct.unpack("<IHHHHIIH", tail[eocd_pos : eocd_pos + 22])
        return fields[6], fields[5]

    _, _, zip64_eocd_offset, _ = struct.unpack("<IIQI", tail[locator_pos : locator_pos + 20])
    record = fetch_range(url, zip64_eocd_offset, zip64_eocd_offset + 120, timeout)
    if not record.startswith(ZIP64_EOCD_SIG):
        raise RuntimeError("Could not find ZIP64 EOCD record.")

    fields = struct.unpack("<IQHHIIQQQQ", record[:56])
    central_dir_size = fields[8]
    central_dir_offset = fields[9]
    return central_dir_offset, central_dir_size


def parse_zip64_extra(extra: bytes, member: ZipMember) -> ZipMember:
    offset = 0
    values: list[int] = []
    while offset + 4 <= len(extra):
        header_id, data_size = struct.unpack("<HH", extra[offset : offset + 4])
        data = extra[offset + 4 : offset + 4 + data_size]
        if header_id == ZIP64_EXTRA_ID:
            values = list(struct.unpack("<" + "Q" * (len(data) // 8), data[: len(data) // 8 * 8]))
            break
        offset += 4 + data_size

    index = 0
    compressed_size = member.compressed_size
    uncompressed_size = member.uncompressed_size
    local_header_offset = member.local_header_offset
    if uncompressed_size == 0xFFFFFFFF:
        uncompressed_size = values[index]
        index += 1
    if compressed_size == 0xFFFFFFFF:
        compressed_size = values[index]
        index += 1
    if local_header_offset == 0xFFFFFFFF:
        local_header_offset = values[index]

    return ZipMember(
        name=member.name,
        compression_method=member.compression_method,
        compressed_size=compressed_size,
        uncompressed_size=uncompressed_size,
        local_header_offset=local_header_offset,
    )


def load_members(url: str, timeout: int) -> dict[str, ZipMember]:
    central_dir_offset, central_dir_size = parse_zip64_directory(url, timeout)
    print(
        f"Fetching central directory: {central_dir_size / 1024 / 1024:.1f} MiB",
        flush=True,
    )
    central_dir = fetch_range(url, central_dir_offset, central_dir_offset + central_dir_size - 1, timeout)

    members: dict[str, ZipMember] = {}
    offset = 0
    while offset < len(central_dir):
        if central_dir[offset : offset + 4] != CENTRAL_DIR_SIG:
            raise RuntimeError(f"Bad central directory signature at byte {offset}.")

        fields = struct.unpack("<IHHHHHHIIIHHHHHII", central_dir[offset : offset + 46])
        compression_method = fields[4]
        compressed_size = fields[8]
        uncompressed_size = fields[9]
        name_len = fields[10]
        extra_len = fields[11]
        comment_len = fields[12]
        local_header_offset = fields[16]

        name_start = offset + 46
        name = central_dir[name_start : name_start + name_len].decode("utf-8")
        extra = central_dir[name_start + name_len : name_start + name_len + extra_len]
        member = ZipMember(name, compression_method, compressed_size, uncompressed_size, local_header_offset)
        members[name] = parse_zip64_extra(extra, member)
        offset = name_start + name_len + extra_len + comment_len

    return members


def extract_member(url: str, member: ZipMember, timeout: int) -> bytes:
    header = fetch_range(url, member.local_header_offset, member.local_header_offset + 30 - 1, timeout)
    if not header.startswith(LOCAL_FILE_SIG):
        raise RuntimeError(f"Bad local file header for {member.name}.")

    fields = struct.unpack("<IHHHHHIIIHH", header)
    name_len = fields[9]
    extra_len = fields[10]
    data_start = member.local_header_offset + 30 + name_len + extra_len
    compressed = fetch_range(url, data_start, data_start + member.compressed_size - 1, timeout)

    if member.compression_method == 0:
        return compressed
    if member.compression_method == 8:
        return zlib.decompress(compressed, -zlib.MAX_WBITS)
    raise RuntimeError(f"Unsupported compression method {member.compression_method} for {member.name}.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--member", action="append", required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    members = load_members(args.url, args.timeout)
    args.output_root.mkdir(parents=True, exist_ok=True)

    for name in args.member:
        if name not in members:
            raise KeyError(f"{name!r} not found in remote archive.")
        data = extract_member(args.url, members[name], args.timeout)
        output_path = args.output_root / name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
        print(f"Wrote {output_path} ({len(data)} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
