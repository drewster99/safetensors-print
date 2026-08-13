"""Parsing and validation of a safetensors file into a fully-described report.

Everything this module produces is a plain fact about the file. It performs no
formatting, so the renderer stays the single place that decides how facts are
displayed, and the reader stays the single place that decides what is true.

Parsing is deliberately tolerant: a file that violates the specification is
still described as completely as possible, with every deviation recorded as an
`Issue` rather than aborting the dump. Only damage that prevents the header from
being read at all raises `SafetensorsFormatError`.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .dtypes import MAX_HEADER_SIZE, DType, dtype_named

HEADER_LENGTH_FIELD_SIZE = 8
"""Bytes occupied by the leading unsigned little-endian 64-bit header length."""

METADATA_KEY = "__metadata__"

ERROR = "error"
WARNING = "warning"


class SafetensorsFormatError(Exception):
    """The file cannot be interpreted as safetensors at all."""


@dataclass(frozen=True)
class Issue:
    """A single deviation from the safetensors specification."""

    severity: str
    message: str


@dataclass(frozen=True)
class FileLayout:
    """Byte ranges of the three top-level regions, so every byte is accounted for."""

    path: str
    total_size: int
    declared_header_size: int
    header_json_begin: int
    header_json_end: int
    data_buffer_begin: int
    data_buffer_size: int

    @property
    def header_length_field_range(self) -> Tuple[int, int]:
        return (0, HEADER_LENGTH_FIELD_SIZE)

    @property
    def data_buffer_end(self) -> int:
        return self.data_buffer_begin + self.data_buffer_size


@dataclass(frozen=True)
class TensorEntry:
    """One tensor as declared in the header, with its derived size arithmetic."""

    name: str
    dtype_name: str
    dtype: Optional[DType]
    shape: Tuple[int, ...]
    begin: int
    end: int
    header_position: int

    @property
    def element_count(self) -> int:
        """Number of elements; an empty shape denotes a scalar, which has one."""
        count = 1
        for dimension in self.shape:
            count *= dimension
        return count

    @property
    def declared_byte_count(self) -> int:
        """Bytes the header's `data_offsets` reserve for this tensor."""
        return self.end - self.begin

    @property
    def expected_bit_count(self) -> Optional[int]:
        if self.dtype is None:
            return None
        return self.element_count * self.dtype.bits_per_element

    @property
    def expected_byte_count(self) -> Optional[int]:
        """Bytes the shape and dtype imply, rounded up when packing leaves a partial byte."""
        bits = self.expected_bit_count
        if bits is None:
            return None
        return (bits + 7) // 8

    @property
    def is_byte_aligned(self) -> bool:
        bits = self.expected_bit_count
        return bits is None or bits % 8 == 0

    @property
    def size_matches_declaration(self) -> Optional[bool]:
        expected = self.expected_byte_count
        if expected is None:
            return None
        return expected == self.declared_byte_count


@dataclass(frozen=True)
class Gap:
    """A run of bytes in the data buffer that no tensor claims."""

    begin: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True)
class Overlap:
    """Two tensors laying claim to the same bytes."""

    earlier_name: str
    later_name: str
    begin: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.begin


@dataclass
class Report:
    """The complete description of one safetensors file."""

    layout: FileLayout
    header_length_field_bytes: bytes
    raw_header_bytes: bytes
    header_padding_bytes: bytes
    header: Dict[str, Any]
    metadata: Dict[str, str]
    non_string_metadata_keys: Tuple[str, ...]
    tensors: List[TensorEntry]
    unparsable_entries: Dict[str, Any]
    duplicate_header_keys: Tuple[str, ...]
    gaps: List[Gap]
    overlaps: List[Overlap]
    issues: List[Issue] = field(default_factory=list)

    @property
    def tensors_in_offset_order(self) -> List[TensorEntry]:
        return sorted(self.tensors, key=lambda tensor: (tensor.begin, tensor.end, tensor.name))

    @property
    def claimed_byte_count(self) -> int:
        """Bytes of the data buffer covered by at least one tensor."""
        return self.layout.data_buffer_size - sum(gap.size for gap in self.gaps)

    @property
    def header_declared_in_offset_order(self) -> bool:
        offsets = [tensor.begin for tensor in self.tensors]
        return offsets == sorted(offsets)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == ERROR for issue in self.issues)


def _duplicate_detecting_object_hook(duplicates: List[str]):
    def hook(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        seen: Dict[str, Any] = {}
        for key, value in pairs:
            if key in seen:
                duplicates.append(key)
            seen[key] = value
        return seen

    return hook


def _read_header_length(file_bytes_prefix: bytes, path: str) -> int:
    if len(file_bytes_prefix) < HEADER_LENGTH_FIELD_SIZE:
        raise SafetensorsFormatError(
            "{}: file is {} bytes, too short to contain the 8-byte header length field".format(
                path, len(file_bytes_prefix)
            )
        )
    return struct.unpack("<Q", file_bytes_prefix[:HEADER_LENGTH_FIELD_SIZE])[0]


def _parse_shape(raw_shape: Any) -> Optional[Tuple[int, ...]]:
    if not isinstance(raw_shape, list):
        return None
    if not all(isinstance(dimension, int) and not isinstance(dimension, bool) for dimension in raw_shape):
        return None
    if any(dimension < 0 for dimension in raw_shape):
        return None
    return tuple(raw_shape)


def _parse_offsets(raw_offsets: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(raw_offsets, list) or len(raw_offsets) != 2:
        return None
    begin, end = raw_offsets
    for value in (begin, end):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return None
    return (begin, end)


def _collect_tensors(
    header: Dict[str, Any], issues: List[Issue]
) -> Tuple[List[TensorEntry], Dict[str, Any]]:
    tensors: List[TensorEntry] = []
    unparsable: Dict[str, Any] = {}

    for position, (name, raw_entry) in enumerate(header.items()):
        if name == METADATA_KEY:
            continue
        if not isinstance(raw_entry, dict):
            unparsable[name] = raw_entry
            issues.append(Issue(ERROR, "tensor {!r}: header entry is not a JSON object".format(name)))
            continue

        dtype_name = raw_entry.get("dtype")
        shape = _parse_shape(raw_entry.get("shape"))
        offsets = _parse_offsets(raw_entry.get("data_offsets"))

        if not isinstance(dtype_name, str) or shape is None or offsets is None:
            unparsable[name] = raw_entry
            issues.append(
                Issue(
                    ERROR,
                    "tensor {!r}: entry is missing or malformed "
                    "'dtype', 'shape' or 'data_offsets'".format(name),
                )
            )
            continue

        dtype = dtype_named(dtype_name)
        if dtype is None:
            issues.append(
                Issue(ERROR, "tensor {!r}: dtype {!r} is not defined by the format".format(name, dtype_name))
            )

        begin, end = offsets
        if end < begin:
            issues.append(
                Issue(
                    ERROR,
                    "tensor {!r}: data_offsets end {} precedes begin {}".format(name, end, begin),
                )
            )

        tensors.append(
            TensorEntry(
                name=name,
                dtype_name=dtype_name,
                dtype=dtype,
                shape=shape,
                begin=begin,
                end=max(begin, end),
                header_position=position,
            )
        )

    return tensors, unparsable


def _coverage(tensors: Sequence[TensorEntry], data_buffer_size: int) -> Tuple[List[Gap], List[Overlap]]:
    """Gaps and overlaps across the data buffer, derived from the tensors' claimed ranges."""
    gaps: List[Gap] = []
    overlaps: List[Overlap] = []

    ordered = sorted(tensors, key=lambda tensor: (tensor.begin, tensor.end, tensor.name))
    highest_claimed_end = 0
    tensor_owning_highest_end: Optional[TensorEntry] = None

    for tensor in ordered:
        if tensor.begin > highest_claimed_end:
            gaps.append(Gap(highest_claimed_end, tensor.begin))
        elif tensor.begin < highest_claimed_end and tensor.declared_byte_count > 0:
            overlaps.append(
                Overlap(
                    earlier_name=tensor_owning_highest_end.name if tensor_owning_highest_end else "<start>",
                    later_name=tensor.name,
                    begin=tensor.begin,
                    end=min(tensor.end, highest_claimed_end),
                )
            )
        if tensor.end > highest_claimed_end:
            highest_claimed_end = tensor.end
            tensor_owning_highest_end = tensor

    if highest_claimed_end < data_buffer_size:
        gaps.append(Gap(highest_claimed_end, data_buffer_size))

    return gaps, overlaps


def _validate_metadata(header: Dict[str, Any], issues: List[Issue]) -> Tuple[Dict[str, str], Tuple[str, ...]]:
    raw_metadata = header.get(METADATA_KEY)
    if raw_metadata is None:
        return {}, ()
    if not isinstance(raw_metadata, dict):
        issues.append(Issue(ERROR, "__metadata__ is present but is not a JSON object"))
        return {}, ()

    metadata: Dict[str, str] = {}
    non_string_keys: List[str] = []
    for key, value in raw_metadata.items():
        if isinstance(value, str):
            metadata[key] = value
        else:
            non_string_keys.append(key)
            metadata[key] = json.dumps(value, sort_keys=True)
            issues.append(
                Issue(
                    ERROR,
                    "__metadata__[{!r}] is {}, but the format requires every metadata "
                    "value to be a string".format(key, type(value).__name__),
                )
            )
    return metadata, tuple(non_string_keys)


def read_report(path: str) -> Report:
    """Parse `path` and return everything the file states about itself.

    Raises `SafetensorsFormatError` when the header cannot be located or decoded,
    since no meaningful description is possible in that case.
    """
    issues: List[Issue] = []

    with open(path, "rb") as handle:
        handle.seek(0, 2)
        total_size = handle.tell()
        handle.seek(0)
        length_field_bytes = handle.read(HEADER_LENGTH_FIELD_SIZE)
        declared_header_size = _read_header_length(length_field_bytes, path)

        # Refusing before the read matters as much as reporting it: the declared size
        # is attacker-controlled, and reading it would allocate that much memory.
        if declared_header_size > MAX_HEADER_SIZE:
            raise SafetensorsFormatError(
                "{}: declared header size {:,} exceeds the {:,}-byte maximum accepted "
                "by the reference implementation".format(path, declared_header_size, MAX_HEADER_SIZE)
            )

        header_json_begin = HEADER_LENGTH_FIELD_SIZE
        header_json_end = header_json_begin + declared_header_size
        if header_json_end > total_size:
            raise SafetensorsFormatError(
                "{}: header claims {:,} bytes ending at offset {:,}, "
                "past the {:,}-byte end of file".format(
                    path, declared_header_size, header_json_end, total_size
                )
            )

        raw_header_bytes = handle.read(declared_header_size)

    if not raw_header_bytes.startswith(b"{"):
        issues.append(
            Issue(ERROR, "header does not begin with '{' (0x7B) as the specification requires")
        )

    # Trailing NUL bytes are stripped alongside whitespace so that a file padded the
    # wrong way is still described, rather than failing as unparsable JSON.
    stripped_header = raw_header_bytes.rstrip(b" \t\r\n\x00")
    header_padding_bytes = raw_header_bytes[len(stripped_header):]
    non_space_padding = set(header_padding_bytes) - {0x20}
    if non_space_padding:
        issues.append(
            Issue(
                ERROR,
                "header padding contains {} but the specification permits only spaces (0x20)".format(
                    ", ".join("0x{:02X}".format(byte) for byte in sorted(non_space_padding))
                ),
            )
        )

    duplicate_keys: List[str] = []
    try:
        header = json.loads(
            stripped_header.decode("utf-8"),
            object_pairs_hook=_duplicate_detecting_object_hook(duplicate_keys),
        )
    except UnicodeDecodeError as error:
        raise SafetensorsFormatError("{}: header is not valid UTF-8: {}".format(path, error))
    except json.JSONDecodeError as error:
        raise SafetensorsFormatError("{}: header is not valid JSON: {}".format(path, error))

    if not isinstance(header, dict):
        raise SafetensorsFormatError(
            "{}: header is a JSON {}, but must be an object".format(path, type(header).__name__)
        )

    for key in duplicate_keys:
        issues.append(
            Issue(ERROR, "header contains duplicate key {!r}; only the last occurrence survives".format(key))
        )

    metadata, non_string_metadata_keys = _validate_metadata(header, issues)
    tensors, unparsable_entries = _collect_tensors(header, issues)

    data_buffer_begin = header_json_end
    data_buffer_size = max(0, total_size - data_buffer_begin)

    for tensor in tensors:
        if tensor.end > data_buffer_size:
            issues.append(
                Issue(
                    ERROR,
                    "tensor {!r}: data_offsets end {:,} lies past the {:,}-byte "
                    "data buffer".format(tensor.name, tensor.end, data_buffer_size),
                )
            )
        if not tensor.is_byte_aligned:
            issues.append(
                Issue(
                    ERROR,
                    "tensor {!r}: {:,} elements of {} occupy {:,} bits, which is not a whole "
                    "number of bytes".format(
                        tensor.name, tensor.element_count, tensor.dtype_name, tensor.expected_bit_count
                    ),
                )
            )
        if tensor.size_matches_declaration is False:
            issues.append(
                Issue(
                    ERROR,
                    "tensor {!r}: shape and dtype imply {:,} bytes but data_offsets "
                    "reserve {:,}".format(tensor.name, tensor.expected_byte_count, tensor.declared_byte_count),
                )
            )

    gaps, overlaps = _coverage(tensors, data_buffer_size)
    for gap in gaps:
        issues.append(
            Issue(
                ERROR,
                "data buffer bytes {:,}..{:,} ({:,} bytes) are claimed by no tensor; "
                "the format forbids holes".format(gap.begin, gap.end, gap.size),
            )
        )
    for overlap in overlaps:
        issues.append(
            Issue(
                ERROR,
                "tensors {!r} and {!r} both claim data buffer bytes {:,}..{:,}".format(
                    overlap.earlier_name, overlap.later_name, overlap.begin, overlap.end
                ),
            )
        )

    report = Report(
        layout=FileLayout(
            path=path,
            total_size=total_size,
            declared_header_size=declared_header_size,
            header_json_begin=header_json_begin,
            header_json_end=header_json_end,
            data_buffer_begin=data_buffer_begin,
            data_buffer_size=data_buffer_size,
        ),
        header_length_field_bytes=length_field_bytes,
        raw_header_bytes=raw_header_bytes,
        header_padding_bytes=header_padding_bytes,
        header=header,
        metadata=metadata,
        non_string_metadata_keys=non_string_metadata_keys,
        tensors=tensors,
        unparsable_entries=unparsable_entries,
        duplicate_header_keys=tuple(duplicate_keys),
        gaps=gaps,
        overlaps=overlaps,
        issues=issues,
    )

    if not report.header_declared_in_offset_order:
        report.issues.append(
            Issue(
                WARNING,
                "tensors are not listed in ascending data_offsets order; the specification "
                "recommends sorting the header by offset",
            )
        )

    return report


class SegmentReader:
    """Reads raw bytes out of a file's data buffer, keeping one handle open.

    Owning the translation from data-buffer offsets to absolute file offsets here
    keeps that arithmetic in a single place.
    """

    def __init__(self, path: str, layout: FileLayout):
        self._layout = layout
        self._handle = open(path, "rb")

    def __enter__(self) -> "SegmentReader":
        return self

    def __exit__(self, *exception_details: Any) -> None:
        self.close()

    def close(self) -> None:
        self._handle.close()

    def read(self, begin: int, byte_count: int) -> bytes:
        """`byte_count` bytes starting at `begin`, measured from the data buffer's start."""
        if byte_count <= 0:
            return b""
        self._handle.seek(self._layout.data_buffer_begin + begin)
        return self._handle.read(byte_count)

    def read_head(self, tensor: TensorEntry, limit: int) -> bytes:
        """Up to `limit` bytes from the start of `tensor`'s region."""
        return self.read(tensor.begin, min(limit, tensor.declared_byte_count))

    def read_tail(self, tensor: TensorEntry, limit: int) -> bytes:
        """Up to `limit` bytes from the end of `tensor`'s region."""
        available = min(limit, tensor.declared_byte_count)
        return self.read(tensor.end - available, available)
