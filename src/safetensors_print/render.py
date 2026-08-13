"""Formatting of a `Report` into the lines the tool prints.

This module decides only how facts look. Every fact it renders is read from the
`Report` the reader produced.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .dtypes import decoder_for
from .reader import (
    METADATA_KEY,
    Gap,
    Report,
    SegmentReader,
    TensorEntry,
)

RULE_WIDTH = 100
PREVIEW_ELEMENT_COUNT = 8
PREVIEW_BYTE_COUNT = 32
HEX_BYTES_PER_LINE = 16

_BINARY_UNITS = ("bytes", "KiB", "MiB", "GiB", "TiB", "PiB")


def human_byte_count(byte_count: int) -> str:
    """A binary-prefixed size, e.g. `33,799,602 bytes (32.24 MiB)`."""
    exact = "{:,} bytes".format(byte_count)
    if byte_count < 1024:
        return exact
    scaled = float(byte_count)
    unit_index = 0
    while scaled >= 1024 and unit_index < len(_BINARY_UNITS) - 1:
        scaled /= 1024
        unit_index += 1
    return "{} ({:.2f} {})".format(exact, scaled, _BINARY_UNITS[unit_index])


def format_shape(shape: Sequence[int]) -> str:
    """A shape as `128x30x7x7`, or `scalar` when the shape is empty."""
    if not shape:
        return "scalar"
    return "x".join(str(dimension) for dimension in shape)


def format_percentage(part: int, whole: int) -> str:
    if whole == 0:
        return "n/a"
    return "{:.4f}%".format(100.0 * part / whole)


def section(title: str) -> Iterator[str]:
    yield ""
    yield "=" * RULE_WIDTH
    yield title
    yield "=" * RULE_WIDTH


def labelled(label: str, value: Any, label_width: int = 30) -> str:
    return "  {}  {}".format(label.ljust(label_width), value)


def hex_dump(raw: bytes, first_offset: int) -> Iterator[str]:
    """Classic offset / hex / printable-ASCII dump of `raw`."""
    for line_start in range(0, len(raw), HEX_BYTES_PER_LINE):
        chunk = raw[line_start : line_start + HEX_BYTES_PER_LINE]
        hex_pairs = " ".join("{:02x}".format(byte) for byte in chunk)
        printable = "".join(chr(byte) if 32 <= byte < 127 else "." for byte in chunk)
        yield "      {:>12,}  {:<{}}  |{}|".format(
            first_offset + line_start, hex_pairs, HEX_BYTES_PER_LINE * 3 - 1, printable
        )


def render_table(headings: Sequence[str], rows: Sequence[Sequence[str]], right_aligned: Sequence[int]) -> Iterator[str]:
    """A fixed-width table sized to its widest cell in each column."""
    if not rows:
        return
    column_count = len(headings)
    widths = [len(heading) for heading in headings]
    for row in rows:
        for column in range(column_count):
            widths[column] = max(widths[column], len(row[column]))

    def format_row(cells: Sequence[str]) -> str:
        formatted = []
        for column, cell in enumerate(cells):
            if column in right_aligned:
                formatted.append(cell.rjust(widths[column]))
            else:
                formatted.append(cell.ljust(widths[column]))
        return "  " + "  ".join(formatted).rstrip()

    yield format_row(headings)
    yield "  " + "  ".join("-" * width for width in widths)
    for row in rows:
        yield format_row(row)


def pretty_header_json(header: Dict[str, Any]) -> str:
    """The header re-serialized with sorted keys and two-space indentation."""
    return json.dumps(header, indent=2, sort_keys=True, ensure_ascii=False)


def _render_file_section(report: Report, verbose: bool) -> Iterator[str]:
    layout = report.layout
    yield from section("FILE")
    yield labelled("Path", layout.path)
    yield labelled("Total size", human_byte_count(layout.total_size))
    yield labelled(
        "Header length field",
        "bytes {:,}..{:,} (8-byte unsigned little-endian) = {:,}".format(
            *layout.header_length_field_range, layout.declared_header_size
        ),
    )
    yield labelled(
        "Header JSON",
        "bytes {:,}..{:,} -- {}".format(
            layout.header_json_begin, layout.header_json_end, human_byte_count(layout.declared_header_size)
        ),
    )
    yield labelled(
        "Data buffer",
        "bytes {:,}..{:,} -- {}".format(
            layout.data_buffer_begin, layout.data_buffer_end, human_byte_count(layout.data_buffer_size)
        ),
    )
    if verbose:
        yield labelled(
            "Header length field bytes",
            " ".join("{:02x}".format(byte) for byte in report.header_length_field_bytes),
        )


def _render_integrity_section(report: Report) -> Iterator[str]:
    layout = report.layout
    yield from section("INTEGRITY")
    yield labelled("Header entries", "{:,}".format(len(report.header)))
    yield labelled("Tensors", "{:,}".format(len(report.tensors)))
    yield labelled("__metadata__ present", "yes ({:,} keys)".format(len(report.metadata)) if METADATA_KEY in report.header else "no")
    yield labelled("Unparsable header entries", "{:,}".format(len(report.unparsable_entries)))
    yield labelled("Duplicate header keys", "{:,}".format(len(report.duplicate_header_keys)))
    yield labelled(
        "Header JSON trailing padding",
        "{:,} bytes{}".format(
            len(report.header_padding_bytes),
            ""
            if not report.header_padding_bytes
            else " ({})".format(" ".join("{:02x}".format(byte) for byte in report.header_padding_bytes[:32])),
        ),
    )
    yield labelled(
        "Data buffer coverage",
        "{:,} of {:,} bytes ({})".format(
            report.claimed_byte_count, layout.data_buffer_size, format_percentage(report.claimed_byte_count, layout.data_buffer_size)
        ),
    )
    yield labelled("Gaps", "none" if not report.gaps else "{:,}".format(len(report.gaps)))
    yield labelled("Overlaps", "none" if not report.overlaps else "{:,}".format(len(report.overlaps)))
    yield labelled(
        "Header sorted by data_offsets",
        "yes" if report.header_declared_in_offset_order else "no (specification recommends sorted)",
    )
    mismatches = sum(1 for tensor in report.tensors if tensor.size_matches_declaration is False)
    yield labelled("Size/shape/dtype agreement", "all {:,} agree".format(len(report.tensors)) if not mismatches else "{:,} disagree".format(mismatches))


def _render_issues_section(report: Report) -> Iterator[str]:
    yield from section("ISSUES")
    if not report.issues:
        yield "  None. The file conforms to the safetensors specification."
        return
    label_width = max(len(issue.severity) for issue in report.issues)
    for issue in report.issues:
        yield "  {}  {}".format(issue.severity.upper().ljust(label_width), issue.message)


def _render_metadata_section(report: Report, verbose: bool) -> Iterator[str]:
    yield from section("__METADATA__")
    if METADATA_KEY not in report.header:
        yield "  The file declares no __metadata__ key."
        return
    if not report.metadata:
        yield "  __metadata__ is present but empty."
        return

    for key in sorted(report.metadata):
        value = report.metadata[key]
        marker = "  (NOT A STRING IN THE FILE)" if key in report.non_string_metadata_keys else ""
        yield "  {}{}".format(key, marker)
        yield "      {}".format(value)
        if verbose:
            yield from _render_nested_json(value)


def _render_nested_json(value: str) -> Iterator[str]:
    """Expand a metadata value that itself holds JSON, which is common for model configs."""
    try:
        nested = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return
    if not isinstance(nested, (dict, list)):
        return
    yield "      -- decoded as JSON --"
    for line in json.dumps(nested, indent=2, sort_keys=True, ensure_ascii=False).splitlines():
        yield "      {}".format(line)


def _render_tensors_section(report: Report) -> Iterator[str]:
    yield from section("TENSORS")
    if not report.tensors:
        yield "  The header declares no tensors."
        return

    rows = []
    for tensor in sorted(report.tensors, key=lambda entry: entry.name):
        expected = tensor.expected_byte_count
        size_note = ""
        if expected is not None and expected != tensor.declared_byte_count:
            size_note = " (expected {:,})".format(expected)
        rows.append(
            [
                tensor.name,
                tensor.dtype_name if tensor.dtype is not None else tensor.dtype_name + " (UNKNOWN)",
                format_shape(tensor.shape),
                "{:,}".format(tensor.element_count),
                "{:,}{}".format(tensor.declared_byte_count, size_note),
                "{:,}..{:,}".format(tensor.begin, tensor.end),
            ]
        )
    yield from render_table(
        ["NAME", "DTYPE", "SHAPE", "ELEMENTS", "BYTES", "DATA_OFFSETS"],
        rows,
        right_aligned=(3, 4, 5),
    )


def _render_dtype_summary(report: Report) -> Iterator[str]:
    yield from section("DTYPE SUMMARY")
    if not report.tensors:
        yield "  The header declares no tensors."
        return

    totals: Dict[str, List[int]] = {}
    for tensor in report.tensors:
        entry = totals.setdefault(tensor.dtype_name, [0, 0, 0])
        entry[0] += 1
        entry[1] += tensor.element_count
        entry[2] += tensor.declared_byte_count

    rows = []
    for dtype_name in sorted(totals):
        tensor_count, element_count, byte_count = totals[dtype_name]
        example = next(tensor for tensor in report.tensors if tensor.dtype_name == dtype_name)
        description = example.dtype.description if example.dtype is not None else "not defined by the format"
        bits = "{}".format(example.dtype.bits_per_element) if example.dtype is not None else "?"
        rows.append(
            [
                dtype_name,
                bits,
                "{:,}".format(tensor_count),
                "{:,}".format(element_count),
                "{:,}".format(byte_count),
                description,
            ]
        )
    yield from render_table(
        ["DTYPE", "BITS", "TENSORS", "ELEMENTS", "BYTES", "DESCRIPTION"],
        rows,
        right_aligned=(1, 2, 3, 4),
    )


def _preview_values(tensor: TensorEntry, raw: bytes) -> Optional[str]:
    """The leading element values of a tensor, when its dtype has an exact Python form."""
    if tensor.dtype is None:
        return None
    decode = decoder_for(tensor.dtype)
    if decode is None:
        return None
    element_size = tensor.dtype.bits_per_element // 8
    if element_size == 0:
        return None
    count = min(PREVIEW_ELEMENT_COUNT, len(raw) // element_size, tensor.element_count)
    if count == 0:
        return None
    values = decode(raw, count)
    rendered = [
        "{:.6g}".format(value) if isinstance(value, float) else str(value) for value in values
    ]
    suffix = ", ..." if tensor.element_count > count else ""
    return "[" + ", ".join(rendered) + suffix + "]"


def _render_segment_map(report: Report, verbose: bool) -> Iterator[str]:
    yield from section("DATA SEGMENT MAP (ordered by offset within the data buffer)")
    if not report.tensors and not report.gaps:
        yield "  The data buffer is empty."
        return

    gaps_by_begin = {gap.begin: gap for gap in report.gaps}
    ordered = report.tensors_in_offset_order

    if not verbose:
        rows: List[List[str]] = []
        cursor = 0
        for tensor in ordered:
            gap = gaps_by_begin.get(cursor)
            if gap is not None and gap.begin < tensor.begin:
                rows.append(["-- GAP --", "", "", "{:,}".format(gap.size), "{:,}..{:,}".format(gap.begin, gap.end)])
                cursor = gap.end
            rows.append(
                [
                    tensor.name,
                    tensor.dtype_name,
                    format_shape(tensor.shape),
                    "{:,}".format(tensor.declared_byte_count),
                    "{:,}..{:,}".format(tensor.begin, tensor.end),
                ]
            )
            cursor = max(cursor, tensor.end)
        trailing = gaps_by_begin.get(cursor)
        if trailing is not None:
            rows.append(
                ["-- GAP --", "", "", "{:,}".format(trailing.size), "{:,}..{:,}".format(trailing.begin, trailing.end)]
            )
        yield from render_table(
            ["SEGMENT", "DTYPE", "SHAPE", "BYTES", "RANGE IN DATA BUFFER"], rows, right_aligned=(3, 4)
        )
        return

    with SegmentReader(report.layout.path, report.layout) as segments:
        cursor = 0
        for tensor in ordered:
            gap = gaps_by_begin.get(cursor)
            if gap is not None and gap.begin < tensor.begin:
                yield from _render_gap(gap, segments)
                cursor = gap.end
            yield ""
            yield "  {}".format(tensor.name)
            yield labelled("dtype", "{} ({} bits per element)".format(tensor.dtype_name, tensor.dtype.bits_per_element if tensor.dtype else "?"), 26)
            yield labelled(
                "shape",
                "{}  ({:,} element{})".format(
                    format_shape(tensor.shape), tensor.element_count, "" if tensor.element_count == 1 else "s"
                ),
                26,
            )
            yield labelled("data_offsets", "{:,}..{:,}".format(tensor.begin, tensor.end), 26)
            yield labelled("absolute file offsets", "{:,}..{:,}".format(
                report.layout.data_buffer_begin + tensor.begin, report.layout.data_buffer_begin + tensor.end
            ), 26)
            yield labelled("declared size", human_byte_count(tensor.declared_byte_count), 26)
            if tensor.expected_byte_count is not None:
                yield labelled(
                    "size from shape/dtype",
                    "{} -- {}".format(
                        human_byte_count(tensor.expected_byte_count),
                        "matches" if tensor.size_matches_declaration else "DOES NOT MATCH",
                    ),
                    26,
                )
            head = segments.read_head(tensor, PREVIEW_BYTE_COUNT)
            preview = _preview_values(tensor, head)
            if preview is not None:
                yield labelled("first elements", preview, 26)
            if head:
                yield "    first {} bytes:".format(len(head))
                yield from hex_dump(head, tensor.begin)
            if tensor.declared_byte_count > PREVIEW_BYTE_COUNT:
                tail = segments.read_tail(tensor, PREVIEW_BYTE_COUNT)
                yield "    last {} bytes:".format(len(tail))
                yield from hex_dump(tail, tensor.end - len(tail))
            cursor = max(cursor, tensor.end)

        trailing = gaps_by_begin.get(cursor)
        if trailing is not None:
            yield from _render_gap(trailing, segments)


def _render_gap(gap: Gap, segments: SegmentReader) -> Iterator[str]:
    yield ""
    yield "  -- GAP: {:,} bytes at {:,}..{:,}, claimed by no tensor --".format(gap.size, gap.begin, gap.end)
    raw = segments.read(gap.begin, min(PREVIEW_BYTE_COUNT, gap.size))
    if raw:
        yield from hex_dump(raw, gap.begin)


def _render_raw_header(report: Report) -> Iterator[str]:
    yield from section("HEADER JSON (pretty-printed, keys sorted)")
    yield pretty_header_json(report.header)


def render_report(report: Report, verbose: bool) -> Iterator[str]:
    """Every line of the default (and, when `verbose`, the expanded) dump."""
    yield from _render_file_section(report, verbose)
    yield from _render_integrity_section(report)
    yield from _render_issues_section(report)
    yield from _render_metadata_section(report, verbose)
    yield from _render_tensors_section(report)
    yield from _render_dtype_summary(report)
    yield from _render_segment_map(report, verbose)
    yield from _render_raw_header(report)
    yield ""
