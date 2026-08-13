"""Parsing and validation behaviour, including every specification violation the reader detects."""

from __future__ import annotations

import struct

import pytest

from safetensors_print.dtypes import MAX_HEADER_SIZE
from safetensors_print.reader import (
    ERROR,
    WARNING,
    SafetensorsFormatError,
    SegmentReader,
    read_report,
)

from .support import build_file_bytes, build_file_bytes_from_raw_header, float32_bytes, write_file


def _messages(report, severity=None):
    return [issue.message for issue in report.issues if severity is None or issue.severity == severity]


def test_minimal_valid_file_has_no_issues(tmp_path):
    header = {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, float32_bytes(1.5, -2.25)))

    report = read_report(path)

    assert report.issues == []
    assert report.has_errors is False
    assert len(report.tensors) == 1
    assert report.tensors[0].element_count == 2
    assert report.tensors[0].declared_byte_count == 8
    assert report.claimed_byte_count == 8


def test_layout_accounts_for_every_byte(tmp_path):
    header = {"w": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}
    raw = build_file_bytes(header, b"\x00\x01\x02\x03")
    path = write_file(tmp_path, "m.safetensors", raw)

    layout = read_report(path).layout

    assert layout.total_size == len(raw)
    assert layout.header_length_field_range == (0, 8)
    assert layout.header_json_begin == 8
    assert layout.declared_header_size == layout.header_json_end - layout.header_json_begin
    assert layout.data_buffer_begin == layout.header_json_end
    assert layout.data_buffer_end == layout.total_size
    assert layout.data_buffer_size == 4


def test_scalar_shape_counts_one_element(tmp_path):
    header = {"s": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]}}
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, float32_bytes(3.0)))

    report = read_report(path)

    assert report.tensors[0].element_count == 1
    assert report.issues == []


def test_zero_element_tensor_is_valid(tmp_path):
    header = {"empty": {"dtype": "F32", "shape": [0, 4], "data_offsets": [0, 0]}}
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, b""))

    report = read_report(path)

    assert report.tensors[0].element_count == 0
    assert report.tensors[0].declared_byte_count == 0
    assert report.issues == []


def test_metadata_is_extracted_and_excluded_from_tensors(tmp_path):
    header = {
        "__metadata__": {"format": "pt", "note": "hello"},
        "w": {"dtype": "I8", "shape": [1], "data_offsets": [0, 1]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, b"\x07"))

    report = read_report(path)

    assert report.metadata == {"format": "pt", "note": "hello"}
    assert [tensor.name for tensor in report.tensors] == ["w"]
    assert report.issues == []


def test_non_string_metadata_value_is_an_error(tmp_path):
    header = {
        "__metadata__": {"steps": 5000},
        "w": {"dtype": "I8", "shape": [1], "data_offsets": [0, 1]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, b"\x07"))

    report = read_report(path)

    assert report.non_string_metadata_keys == ("steps",)
    assert report.metadata["steps"] == "5000"
    assert any("must be a string" in message or "requires every metadata" in message for message in _messages(report, ERROR))


def test_duplicate_header_keys_are_reported(tmp_path):
    """A warning, not an error: JSON says names SHOULD be unique, and readers cope.

    The reference implementation loads such a file, keeping the last occurrence, so
    failing it here would make the exit code disagree with every other reader.
    """
    raw_header = (
        b'{"w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},'
        b' "w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}'
    )
    path = write_file(tmp_path, "m.safetensors", build_file_bytes_from_raw_header(raw_header, b"\x00"))

    report = read_report(path)

    assert report.duplicate_header_keys == ("w",)
    assert any("duplicate key" in message for message in _messages(report, WARNING))
    assert not report.has_errors


def test_gap_in_data_buffer_is_reported(tmp_path):
    header = {
        "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
        "b": {"dtype": "U8", "shape": [2], "data_offsets": [6, 8]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(8)))

    report = read_report(path)

    assert [(gap.begin, gap.end) for gap in report.gaps] == [(2, 6)]
    assert report.claimed_byte_count == 4
    assert any("claimed by no tensor" in message for message in _messages(report, ERROR))


def test_trailing_gap_is_reported(tmp_path):
    header = {"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}}
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(10)))

    report = read_report(path)

    assert [(gap.begin, gap.end) for gap in report.gaps] == [(2, 10)]


def test_overlapping_tensors_are_reported(tmp_path):
    header = {
        "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
        "b": {"dtype": "U8", "shape": [4], "data_offsets": [2, 6]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(6)))

    report = read_report(path)

    assert len(report.overlaps) == 1
    assert (report.overlaps[0].begin, report.overlaps[0].end) == (2, 4)
    assert any("both claim data buffer bytes" in message for message in _messages(report, ERROR))


def test_unknown_dtype_is_reported_without_aborting(tmp_path):
    header = {"w": {"dtype": "F128", "shape": [1], "data_offsets": [0, 16]}}
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(16)))

    report = read_report(path)

    assert report.tensors[0].dtype is None
    assert report.tensors[0].expected_byte_count is None
    assert report.tensors[0].size_matches_declaration is None
    assert any("not defined by the format" in message for message in _messages(report, ERROR))


def test_size_mismatch_between_shape_and_offsets_is_reported(tmp_path):
    header = {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 8]}}
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(8)))

    report = read_report(path)

    assert report.tensors[0].expected_byte_count == 16
    assert report.tensors[0].size_matches_declaration is False
    assert any("data_offsets" in message and "reserve" in message for message in _messages(report, ERROR))


def test_sub_byte_dtype_packs_two_elements_per_byte(tmp_path):
    header = {"w": {"dtype": "F4", "shape": [4], "data_offsets": [0, 2]}}
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(2)))

    report = read_report(path)

    assert report.tensors[0].dtype.is_sub_byte is True
    assert report.tensors[0].expected_bit_count == 16
    assert report.tensors[0].expected_byte_count == 2
    assert report.issues == []


def test_sub_byte_dtype_with_misaligned_element_count_is_reported(tmp_path):
    header = {"w": {"dtype": "F4", "shape": [3], "data_offsets": [0, 2]}}
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(2)))

    report = read_report(path)

    assert report.tensors[0].is_byte_aligned is False
    assert any("not a whole number of bytes" in message for message in _messages(report, ERROR))


def test_offsets_past_end_of_data_buffer_are_reported(tmp_path):
    header = {"w": {"dtype": "U8", "shape": [16], "data_offsets": [0, 16]}}
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(4)))

    report = read_report(path)

    assert any("lies past the" in message for message in _messages(report, ERROR))


def test_unsorted_offsets_produce_a_warning_not_an_error(tmp_path):
    header = {
        "b": {"dtype": "U8", "shape": [2], "data_offsets": [2, 4]},
        "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(4)))

    report = read_report(path)

    assert report.header_declared_in_offset_order is False
    assert report.has_errors is False
    assert any("ascending data_offsets order" in message for message in _messages(report, WARNING))
    assert [tensor.name for tensor in report.tensors_in_offset_order] == ["a", "b"]


def test_header_padding_is_measured(tmp_path):
    header = {"w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, b"\x00", padding=b"    "))

    report = read_report(path)

    assert report.header_padding_bytes == b"    "
    assert report.issues == []


def test_malformed_tensor_entry_is_collected_not_fatal(tmp_path):
    raw_header = b'{"good": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}, "bad": 42}'
    path = write_file(tmp_path, "m.safetensors", build_file_bytes_from_raw_header(raw_header, b"\x00"))

    report = read_report(path)

    assert list(report.unparsable_entries) == ["bad"]
    assert [tensor.name for tensor in report.tensors] == ["good"]
    assert any("not a JSON object" in message for message in _messages(report, ERROR))


def test_entry_missing_required_fields_is_collected(tmp_path):
    header = {"w": {"dtype": "U8", "shape": [1]}}
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, b"\x00"))

    report = read_report(path)

    assert list(report.unparsable_entries) == ["w"]
    assert any("data_offsets" in message for message in _messages(report, ERROR))


def test_negative_shape_dimension_is_rejected(tmp_path):
    header = {"w": {"dtype": "U8", "shape": [-1], "data_offsets": [0, 1]}}
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, b"\x00"))

    report = read_report(path)

    assert list(report.unparsable_entries) == ["w"]


def test_boolean_shape_dimension_is_rejected(tmp_path):
    raw_header = b'{"w": {"dtype": "U8", "shape": [true], "data_offsets": [0, 1]}}'
    path = write_file(tmp_path, "m.safetensors", build_file_bytes_from_raw_header(raw_header, b"\x00"))

    report = read_report(path)

    assert list(report.unparsable_entries) == ["w"]


def test_header_not_beginning_with_brace_is_reported(tmp_path):
    raw_header = b' {"w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}'
    path = write_file(tmp_path, "m.safetensors", build_file_bytes_from_raw_header(raw_header, b"\x00"))

    report = read_report(path)

    assert any("0x7B" in message for message in _messages(report, ERROR))


def test_file_shorter_than_length_field_raises(tmp_path):
    path = write_file(tmp_path, "m.safetensors", b"\x01\x02\x03")

    with pytest.raises(SafetensorsFormatError, match="too short"):
        read_report(path)


def test_header_extending_past_end_of_file_raises(tmp_path):
    path = write_file(tmp_path, "m.safetensors", struct.pack("<Q", 4096) + b"{}")

    with pytest.raises(SafetensorsFormatError, match="past the"):
        read_report(path)


def test_invalid_json_header_raises(tmp_path):
    path = write_file(tmp_path, "m.safetensors", build_file_bytes_from_raw_header(b"{not json}"))

    with pytest.raises(SafetensorsFormatError, match="not valid JSON"):
        read_report(path)


def test_non_utf8_header_raises(tmp_path):
    path = write_file(tmp_path, "m.safetensors", build_file_bytes_from_raw_header(b'{"\xff\xfe": 1}'))

    with pytest.raises(SafetensorsFormatError, match="not valid UTF-8"):
        read_report(path)


def test_json_array_header_raises(tmp_path):
    path = write_file(tmp_path, "m.safetensors", build_file_bytes_from_raw_header(b"[]"))

    with pytest.raises(SafetensorsFormatError, match="must be an object"):
        read_report(path)


def test_header_larger_than_maximum_is_refused_before_it_is_read(tmp_path):
    """The declared size is untrusted input, so an oversized header is refused, not allocated."""
    oversized = MAX_HEADER_SIZE + 1
    path = tmp_path / "m.safetensors"
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", oversized))
        handle.write(b'{"__metadata__": {"note": "oversized header"}}')
        handle.truncate(8 + oversized)  # sparse padding, so no real bytes are written

    with pytest.raises(SafetensorsFormatError, match="exceeds the"):
        read_report(str(path))


def test_header_at_exactly_the_maximum_size_is_accepted(tmp_path):
    """The limit is inclusive, so a header of exactly the maximum size must still parse."""
    header_bytes = b'{"__metadata__": {"note": "at the limit"}}'
    padding = b" " * (MAX_HEADER_SIZE - len(header_bytes))
    path = write_file(
        tmp_path, "m.safetensors", build_file_bytes_from_raw_header(header_bytes + padding)
    )

    report = read_report(path)

    assert report.layout.declared_header_size == MAX_HEADER_SIZE
    assert report.metadata == {"note": "at the limit"}
    assert report.issues == []


def test_file_containing_only_the_length_field_raises(tmp_path):
    path = write_file(tmp_path, "m.safetensors", struct.pack("<Q", 0))

    with pytest.raises(SafetensorsFormatError, match="not valid JSON"):
        read_report(path)


def test_nul_padded_header_is_parsed_and_the_padding_is_reported(tmp_path):
    """Padding must be spaces, but a NUL-padded header is still fully described."""
    header = {"w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, b"\x00", padding=b"\x00\x00"))

    report = read_report(path)

    assert report.header_padding_bytes == b"\x00\x00"
    assert [tensor.name for tensor in report.tensors] == ["w"]
    assert any("0x00" in message and "only spaces" in message for message in _messages(report, ERROR))


def test_empty_data_buffer_with_no_tensors_has_no_gaps(tmp_path):
    path = write_file(tmp_path, "m.safetensors", build_file_bytes({"__metadata__": {"a": "b"}}))

    report = read_report(path)

    assert report.tensors == []
    assert report.gaps == []
    assert report.issues == []


def test_segment_reader_returns_the_tensor_bytes(tmp_path):
    header = {
        "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
        "b": {"dtype": "U8", "shape": [4], "data_offsets": [4, 8]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(range(8))))

    report = read_report(path)
    by_name = {tensor.name: tensor for tensor in report.tensors}
    with SegmentReader(path, report.layout) as segments:
        assert segments.read_head(by_name["a"], 4) == b"\x00\x01\x02\x03"
        assert segments.read_head(by_name["b"], 4) == b"\x04\x05\x06\x07"
        assert segments.read_tail(by_name["b"], 2) == b"\x06\x07"
        assert segments.read_head(by_name["a"], 100) == b"\x00\x01\x02\x03"


def test_reversed_offsets_are_reported(tmp_path):
    header = {"w": {"dtype": "U8", "shape": [0], "data_offsets": [8, 4]}}
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(8)))

    report = read_report(path)

    assert any("precedes begin" in message for message in _messages(report, ERROR))
