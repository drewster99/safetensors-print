"""Helpers for synthesizing safetensors files, including deliberately malformed ones."""

from __future__ import annotations

import json
import struct
from typing import Any, Dict, Optional


def build_file_bytes(header: Dict[str, Any], data: bytes = b"", padding: bytes = b"") -> bytes:
    """A well-formed file whose header is `header` and whose data buffer is `data`."""
    return build_file_bytes_from_raw_header(
        json.dumps(header).encode("utf-8") + padding, data
    )


def build_file_bytes_from_raw_header(
    header_bytes: bytes, data: bytes = b"", declared_size: Optional[int] = None
) -> bytes:
    """A file with an arbitrary header payload, so invalid headers can be exercised."""
    size = len(header_bytes) if declared_size is None else declared_size
    return struct.pack("<Q", size) + header_bytes + data


def write_file(tmp_path, name: str, content: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(content)
    return str(path)


def float32_bytes(*values: float) -> bytes:
    return struct.pack("<{}f".format(len(values)), *values)
