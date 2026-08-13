#!/usr/bin/env python3
"""Write .safetensors files that exercise the edge cases real files rarely reach.

Downloaded models are all well formed, which is exactly why they cannot test the
half of the tool that reports damage. These files are hand-built to sit on each
boundary: every dtype the format defines, every violation the reader reports, and
every way a header can be too broken to read at all.

    python3 scripts/make-synthetic-corpus.py [directory]

Rewrites the directory from scratch each run, so it is safe to repeat.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
from typing import Any, Dict

DEFAULT_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "corpus", "synthetic"
)

# Every dtype the format defines, with the bits each element occupies.
DTYPE_BITS = {
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "C64": 64,
    "F64": 64,
    "I64": 64,
    "U64": 64,
}


def file_bytes(header: Any, data: bytes, padding: bytes = b"") -> bytes:
    """A whole file: the 8-byte length, the header, its padding, then the data."""
    encoded = json.dumps(header).encode("utf-8") + padding
    return struct.pack("<Q", len(encoded)) + encoded + data


def entry(dtype: str, shape, begin: int, end: int) -> Dict[str, Any]:
    return {"dtype": dtype, "shape": list(shape), "data_offsets": [begin, end]}


def every_dtype() -> bytes:
    """One tensor per dtype, packed end to end, each holding whole bytes."""
    header: Dict[str, Any] = {"__metadata__": {"purpose": "one tensor of every defined dtype"}}
    offset = 0
    for name, bits in DTYPE_BITS.items():
        elements = 8  # Divisible by 2 and 4, so the sub-byte dtypes stay whole-byte.
        size = elements * bits // 8
        header[name.lower()] = entry(name, [elements], offset, offset + size)
        offset += size
    # The buffer must cover every byte claimed above: a short one would make this a
    # file with a hole in it, which is a different case, tested separately.
    return file_bytes(header, bytes(index % 256 for index in range(offset)))


def deeply_encoded() -> bytes:
    """A configuration stored as a JSON-encoded string, the way trainers write them."""
    configuration = {
        "layers": [{"channels": 128, "kernel": 7, "norm": "layer_norm"} for _ in range(6)],
        "activation": "relu",
        "nested": {"deeper": {"deepest": [1, 2, 3, {"leaf": True}]}},
    }
    header = {
        "__metadata__": {
            "architecture": json.dumps(configuration),
            "doubly_encoded": json.dumps({"inner": json.dumps({"leaf": 1})}),
            "an_encoded_array": json.dumps([1, 2, 3]),
            "an_encoded_empty_object": "{}",
            "training_step": "5000",
            "a_plain_string": "not JSON at all",
        },
        "w": entry("F32", [2], 0, 8),
    }
    return file_bytes(header, bytes(8))


def forged_expansion_markers() -> bytes:
    """Metadata spelling out the renderer's own internal marker.

    A JSON string may hold any character, NUL included, so nothing stops a file from
    containing whatever text the renderer uses to keep its place.
    """
    marker = "\x00safetensors-print-expansion:{}\x00"
    header = {
        "__metadata__": {
            "forged_first": marker.format(0),
            "forged_past_the_end": marker.format(9999),
            "forged_inside_encoded": json.dumps({"nested": marker.format(0)}),
            "real": json.dumps({"a": 1}),
        },
        "w": entry("F32", [2], 0, 8),
    }
    return file_bytes(header, bytes(8))


def unicode_metadata() -> bytes:
    header = {
        "__metadata__": {
            "emoji": "♞ knight ♞",
            "accents": "café naïve",
            "cjk": "モデル",
            "quotes": 'he said "hi" \\ and left',
            "newlines": "line one\nline two\ttabbed",
            "encoded_unicode": json.dumps({"名前": "モデル", "emoji": "🚀"}),
        },
        "w": entry("F32", [2], 0, 8),
    }
    return file_bytes(header, bytes(8))


def wide_and_deep() -> bytes:
    """Many tensors with long names, to stress the table's column sizing."""
    header: Dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for index in range(200):
        name = "encoder.layers.{}.self_attention.query_key_value.weight".format(index)
        size = 4 * (index + 1)
        header[name] = entry("F32", [index + 1], offset, offset + size)
        offset += size
    return file_bytes(header, bytes(offset))


CASES = {
    # Conforming files.
    "conforming-minimal": lambda: file_bytes({"w": entry("U8", [1], 0, 1)}, b"\x00"),
    "conforming-no-metadata": lambda: file_bytes(
        {"a": entry("F32", [2], 0, 8), "b": entry("I64", [1], 8, 16)}, bytes(16)
    ),
    "conforming-empty-metadata": lambda: file_bytes(
        {"__metadata__": {}, "w": entry("U8", [1], 0, 1)}, b"\x00"
    ),
    "conforming-no-tensors": lambda: file_bytes({"__metadata__": {"note": "header only"}}, b""),
    "conforming-empty-header": lambda: file_bytes({}, b""),
    "conforming-scalar": lambda: file_bytes(
        {"scalar": entry("F32", [], 0, 4), "empty": entry("F32", [0], 4, 4)}, bytes(4)
    ),
    "conforming-every-dtype": every_dtype,
    "conforming-encoded-metadata": deeply_encoded,
    "conforming-forged-markers": forged_expansion_markers,
    "conforming-unicode-metadata": unicode_metadata,
    "conforming-many-tensors": wide_and_deep,
    "conforming-padded-header": lambda: file_bytes(
        {"w": entry("U8", [1], 0, 1)}, b"\x00", padding=b"   "
    ),
    "warning-unsorted-offsets": lambda: file_bytes(
        {"b": entry("U8", [1], 1, 2), "a": entry("U8", [1], 0, 1)}, bytes(2)
    ),
    # Files that violate the specification but can still be described.
    "violation-gap": lambda: file_bytes({"a": entry("U8", [1], 0, 1)}, bytes(8)),
    "violation-overlap": lambda: file_bytes(
        {"a": entry("U8", [4], 0, 4), "b": entry("U8", [4], 2, 6)}, bytes(6)
    ),
    "violation-unknown-dtype": lambda: file_bytes(
        {"known": entry("U8", [1], 0, 1), "mystery": entry("F128", [1], 1, 17)}, bytes(17)
    ),
    "violation-size-disagrees": lambda: file_bytes({"w": entry("F32", [4], 0, 8)}, bytes(8)),
    "violation-non-string-metadata": lambda: file_bytes(
        {"__metadata__": {"step": 5000, "flag": True}, "w": entry("U8", [1], 0, 1)}, b"\x00"
    ),
    "violation-metadata-not-object": lambda: file_bytes(
        {"__metadata__": [1, 2, 3], "w": entry("U8", [1], 0, 1)}, b"\x00"
    ),
    "violation-metadata-null": lambda: file_bytes(
        {"__metadata__": None, "w": entry("U8", [1], 0, 1)}, b"\x00"
    ),
    "violation-negative-shape": lambda: file_bytes({"w": entry("F32", [-2], 0, 8)}, bytes(8)),
    "violation-reversed-offsets": lambda: file_bytes({"w": entry("U8", [1], 4, 2)}, bytes(8)),
    "violation-offsets-past-buffer": lambda: file_bytes({"w": entry("U8", [64], 0, 64)}, bytes(8)),
    "violation-missing-dtype": lambda: file_bytes(
        {"w": {"shape": [1], "data_offsets": [0, 1]}}, b"\x00"
    ),
    "violation-subbyte-part-byte": lambda: file_bytes({"w": entry("F4", [3], 0, 2)}, bytes(2)),
    "violation-bad-padding": lambda: file_bytes(
        {"w": entry("U8", [1], 0, 1)}, b"\x00", padding=b"\x00\x00"
    ),
    # Written by hand because json.dumps cannot produce a duplicate key. The length is
    # measured rather than counted, since a wrong one would test truncation instead.
    "violation-duplicate-keys": lambda: file_bytes_raw(
        b'{"w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},'
        b' "w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}'
    )
    + b"\x00",
    # Files whose header cannot be read at all.
    "unreadable-empty": lambda: b"",
    "unreadable-truncated-length": lambda: b"\x01\x02\x03",
    "unreadable-header-past-eof": lambda: struct.pack("<Q", 4096) + b'{"a": 1}',
    "unreadable-not-json": lambda: file_bytes_raw(b"this is not JSON at all, not even close"),
    "unreadable-json-not-object": lambda: file_bytes_raw(b"[1, 2, 3]"),
    "unreadable-not-utf8": lambda: file_bytes_raw(b'{"\xff\xfe": 1}'),
    "unreadable-oversized-header": lambda: struct.pack("<Q", 200 * 1000 * 1000) + b"{}",
}


def file_bytes_raw(header: bytes) -> bytes:
    return struct.pack("<Q", len(header)) + header


def main(argv):
    directory = argv[1] if len(argv) > 1 else DEFAULT_DIRECTORY
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    for name in sorted(CASES):
        path = os.path.join(directory, name + ".safetensors")
        with open(path, "wb") as handle:
            handle.write(CASES[name]())
        print("{:>10,}  {}".format(os.path.getsize(path), os.path.basename(path)))

    print("\n{} files in {}".format(len(CASES), directory))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
