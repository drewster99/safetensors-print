"""Formatting of a `Report` into the lines the tool prints.

This module decides only how facts look. Every fact it renders is read from the
`Report` the reader produced.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Sequence, Tuple

from .dtypes import decoder_for
from .reader import (
    Gap,
    MetadataDeclaration,
    Report,
    SegmentReader,
    TensorEntry,
)

RULE_WIDTH = 100
PREVIEW_ELEMENT_COUNT = 8
PREVIEW_BYTE_COUNT = 32
HEX_BYTES_PER_LINE = 16

SORT_BY_OFFSET = "offset"
SORT_BY_NAME = "name"
SORT_ORDERS = (SORT_BY_OFFSET, SORT_BY_NAME)


class Section(Enum):
    """A selectable part of the dump.

    `SUMMARY` covers the file at a glance: its layout, the integrity checks, and the
    per-dtype totals. `METADATA` is the only member the command line cannot select on
    its own, since --metadata prints the same content as JSON a pipeline can consume.
    """

    SUMMARY = "summary"
    ISSUES = "issues"
    METADATA = "metadata"
    TENSORS = "tensors"
    HEADER = "header"


ALL_SECTIONS = frozenset(Section)

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
    """A header or metadata object re-serialized verbatim, sorted, indented by two.

    Strictly valid JSON, byte-faithful to what the file holds. This is what
    `--metadata-raw` prints, and it is the escape hatch for anything that needs to
    consume the file's own bytes rather than read them.
    """
    return json.dumps(header, indent=2, sort_keys=True, ensure_ascii=False)


class ExpansionStyle(Enum):
    """Whether an expanded value is marked as having been stored encoded.

    The mark is a `/* */` comment, which JSON has no syntax for, so it is available
    only where the output is prose for a reader. Anything that must parse as JSON
    takes `PLAIN`, at the cost of no longer showing how the file stores the value.
    """

    ANNOTATED = "annotated"
    PLAIN = "plain"


_EXPANSION_PREFIX = "safetensors-print-expansion:"
_EXPANSION_NOTE = "  /* stored as a JSON-encoded string, shown decoded */"


def _expansion_line_pattern(prefix: str) -> "re.Pattern":
    """Matches a line holding nothing but one marker, optionally behind its key.

    json.dumps escapes NUL as `\\u0000` even with ensure_ascii=False, so a NUL-delimited
    marker survives serialization as a predictable literal.
    """
    return re.compile(
        r'^(?P<head>\s*(?:"(?:[^"\\]|\\.)*"\s*:\s*)?)'
        r'"\\u0000' + re.escape(prefix) + r'(?P<index>\d+)\\u0000"'
        r"(?P<tail>,?)$"
    )


def _substitute_encoded_json(value: Any, prefix: str) -> Tuple[Any, List[Any], int]:
    """`value` with each JSON-holding string swapped for a marker.

    The third result is how many `-` the file's own strings manage to append directly
    after `prefix`, or -1 if none of them contain it at all. That is what it takes to
    choose a prefix the file cannot spell, since a string may hold any text we might
    have picked, NUL included.
    """
    expansions: List[Any] = []
    forgery = re.compile(re.escape(prefix) + "(-*)")
    longest_forgery = -1

    def substitute(node: Any) -> Any:
        nonlocal longest_forgery
        if isinstance(node, dict):
            return {key: substitute(item) for key, item in node.items()}
        if isinstance(node, list):
            return [substitute(item) for item in node]
        if isinstance(node, str):
            for dashes in forgery.findall(node):
                longest_forgery = max(longest_forgery, len(dashes))
            decoded = _decoded_json_value(node)
            if decoded is not None:
                expansions.append(decoded)
                return "\x00{}{}\x00".format(prefix, len(expansions) - 1)
        return node

    return substitute(value), expansions, longest_forgery


def pretty_json_lines(value: Any, style: ExpansionStyle) -> Iterator[str]:
    """Pretty-printed JSON with sorted keys, expanding values that themselves hold JSON.

    A string value holding a JSON object or array is expanded in place rather than
    printed as one escaped line hundreds of characters wide. `style` decides whether
    each expansion is marked as such. Either way the expansion is no longer
    byte-faithful: what the file holds as a string is shown as structure, and
    `pretty_header_json` is the form that reproduces the file.
    """
    # A file may legally contain the marker text: a JSON string can hold anything, NUL
    # included, and a marker the file can spell would be rendered as someone else's
    # value. One more `-` than the file manages is a prefix it cannot have spelled, so
    # this settles in a second pass however hostile the file is.
    prefix = _EXPANSION_PREFIX
    substituted, expansions, longest_forgery = _substitute_encoded_json(value, prefix)
    if longest_forgery >= 0:
        prefix += "-" * (longest_forgery + 1)
        substituted, expansions, _ = _substitute_encoded_json(value, prefix)

    note = _EXPANSION_NOTE if style is ExpansionStyle.ANNOTATED else ""
    pattern = _expansion_line_pattern(prefix)
    serialized = json.dumps(substituted, indent=2, sort_keys=True, ensure_ascii=False)

    for line in serialized.splitlines():
        match = pattern.match(line)
        if match is None:
            yield line
            continue
        head, tail = match.group("head"), match.group("tail")
        indent = head[: len(head) - len(head.lstrip())]
        block = json.dumps(
            expansions[int(match.group("index"))], indent=2, sort_keys=True, ensure_ascii=False
        ).splitlines()
        if len(block) == 1:
            yield head + block[0] + tail + note
            continue
        yield head + block[0] + note
        for nested in block[1:-1]:
            yield indent + nested
        yield indent + block[-1] + tail


def expanded_json_text(value: Any) -> str:
    """`value` pretty-printed with JSON-encoded string values expanded in place.

    Strictly valid JSON, so it can be piped onward, but no longer byte-faithful: a
    value the file stores as an encoded string is shown as the structure it holds.
    `pretty_header_json` is the faithful form.
    """
    return "\n".join(pretty_json_lines(value, ExpansionStyle.PLAIN))


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
    if report.metadata_declaration is MetadataDeclaration.PRESENT:
        metadata_state = "yes ({:,} keys)".format(len(report.metadata))
    elif report.metadata_declaration is MetadataDeclaration.ABSENT:
        metadata_state = "no"
    else:
        metadata_state = report.metadata_declaration.value
    yield labelled("__metadata__ present", metadata_state)
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
    # An unknown dtype leaves the expected size unknowable, which is neither agreement
    # nor disagreement and must not be silently counted as either.
    agreeing = sum(1 for tensor in report.tensors if tensor.size_matches_declaration is True)
    disagreeing = sum(1 for tensor in report.tensors if tensor.size_matches_declaration is False)
    undeterminable = sum(1 for tensor in report.tensors if tensor.size_matches_declaration is None)
    counts = ["{:,} agree".format(agreeing)]
    if disagreeing:
        counts.append("{:,} disagree".format(disagreeing))
    if undeterminable:
        counts.append("{:,} undeterminable (dtype not defined by the format)".format(undeterminable))
    yield labelled("Size/shape/dtype agreement", ", ".join(counts))


def _render_issues_section(report: Report) -> Iterator[str]:
    yield from section("ISSUES")
    if not report.issues:
        yield "  None. The file conforms to the safetensors specification."
        return
    label_width = max(len(issue.severity) for issue in report.issues)
    for issue in report.issues:
        yield "  {}  {}".format(issue.severity.upper().ljust(label_width), issue.message)


# Why there is no metadata to print, said the same way wherever it is said. A
# declaration absent from this mapping is one that has metadata to show.
_NOTHING_TO_SHOW = {
    MetadataDeclaration.ABSENT: "The file declares no __metadata__ key.",
    MetadataDeclaration.NULL: "__metadata__ is declared as null, which readers treat as absent.",
    MetadataDeclaration.NOT_AN_OBJECT: "__metadata__ is present but is not a JSON object.",
    MetadataDeclaration.EMPTY: "__metadata__ is present but empty.",
}


def nothing_to_show_explanation(declaration: MetadataDeclaration) -> Optional[str]:
    """Why a file has no metadata to print, or None when it has some."""
    return _NOTHING_TO_SHOW.get(declaration)


def _render_metadata_section(report: Report) -> Iterator[str]:
    yield from section("__METADATA__")
    explanation = nothing_to_show_explanation(report.metadata_declaration)
    if explanation is not None:
        yield "  " + explanation
        return
    yield from _render_metadata_json(report)


def _decoded_json_value(value: str) -> Optional[Any]:
    """A metadata value's own JSON content, when it holds a JSON object or array.

    Model configurations are routinely stored this way, since the format permits only
    string values. Bare JSON scalars are left alone: `"1000000"` is a string in the
    file and showing it as a number would misrepresent it.
    """
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return None
    return decoded if isinstance(decoded, (dict, list)) else None


def _render_metadata_json(report: Report) -> Iterator[str]:
    if report.non_string_metadata_keys:
        yield "  Values of {} are not strings in the file, which the format requires.".format(
            ", ".join(repr(key) for key in report.non_string_metadata_keys)
        )
    yield from pretty_json_lines(report.metadata, ExpansionStyle.ANNOTATED)


def _tensor_row(tensor: TensorEntry) -> List[str]:
    expected = tensor.expected_byte_count
    size_note = ""
    if expected is not None and expected != tensor.declared_byte_count:
        size_note = " (expected {:,})".format(expected)
    return [
        tensor.name,
        tensor.dtype_name if tensor.dtype is not None else tensor.dtype_name + " (UNKNOWN)",
        format_shape(tensor.shape),
        "{:,}".format(tensor.element_count),
        "{:,}{}".format(tensor.declared_byte_count, size_note),
        "{:,}".format(tensor.begin),
        "{:,}".format(tensor.end),
    ]


def _gap_row(gap: Gap) -> List[str]:
    return [
        "-- unclaimed gap --",
        "",
        "",
        "",
        "{:,}".format(gap.size),
        "{:,}".format(gap.begin),
        "{:,}".format(gap.end),
    ]


def _render_tensors_section(report: Report, sort_by: str) -> Iterator[str]:
    """The one table of tensors. Ordered by offset it doubles as the data buffer's map."""
    if sort_by == SORT_BY_OFFSET:
        yield from section("TENSORS (ordered by offset within the data buffer)")
    else:
        yield from section("TENSORS (ordered by name)")

    if sort_by == SORT_BY_OFFSET:
        rows = [
            _gap_row(item) if isinstance(item, Gap) else _tensor_row(item)
            for item in report.data_buffer_walk()
        ]
    else:
        rows = [_tensor_row(tensor) for tensor in sorted(report.tensors, key=lambda entry: entry.name)]

    # Ordering by name drops the gaps, so a file whose only entries are unusable leaves
    # nothing to tabulate. An empty section under a heading would read as a rendering fault.
    if not rows:
        yield "  The header declares no tensors."
        return

    yield from render_table(
        ["NAME", "DTYPE", "SHAPE", "ELEMENTS", "BYTES", "OFFSET_BEGIN", "OFFSET_END"],
        rows,
        right_aligned=(3, 4, 5, 6),
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


def _render_tensor_detail(report: Report) -> Iterator[str]:
    """Byte-level detail for each segment, in the order it occupies the data buffer."""
    yield from section("TENSOR DETAIL (ordered by offset within the data buffer)")
    if not report.tensors and not report.gaps:
        yield "  The data buffer is empty."
        return

    with SegmentReader(report.layout.path, report.layout) as segments:
        for item in report.data_buffer_walk():
            if isinstance(item, Gap):
                yield from _render_gap(item, segments)
                continue
            tensor = item
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


def _render_gap(gap: Gap, segments: SegmentReader) -> Iterator[str]:
    yield ""
    yield "  -- unclaimed gap: {:,} bytes at {:,}..{:,}, claimed by no tensor --".format(
        gap.size, gap.begin, gap.end
    )
    raw = segments.read(gap.begin, min(PREVIEW_BYTE_COUNT, gap.size))
    if raw:
        yield from hex_dump(raw, gap.begin)


def _render_raw_header(report: Report) -> Iterator[str]:
    yield from section("HEADER JSON (pretty-printed, keys sorted)")
    yield from pretty_json_lines(report.header, ExpansionStyle.ANNOTATED)


def render_report(
    report: Report,
    sections: FrozenSet[Section],
    verbose: bool,
    sort_by: str = SORT_BY_OFFSET,
) -> Iterator[str]:
    """The selected sections of the dump, always in the order the full dump uses.

    `verbose` adds the tensor detail, which belongs to `Section.TENSORS` and so is
    withheld when the tensors were not selected.
    """
    if Section.SUMMARY in sections:
        yield from _render_file_section(report, verbose)
        yield from _render_integrity_section(report)
    if Section.ISSUES in sections:
        yield from _render_issues_section(report)
    if Section.METADATA in sections:
        yield from _render_metadata_section(report)
    if Section.TENSORS in sections:
        yield from _render_tensors_section(report, sort_by)
    if Section.SUMMARY in sections:
        yield from _render_dtype_summary(report)
    if verbose and Section.TENSORS in sections:
        yield from _render_tensor_detail(report)
    if Section.HEADER in sections:
        yield from _render_raw_header(report)
    yield ""
