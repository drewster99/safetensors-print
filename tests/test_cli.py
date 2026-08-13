"""Command line behaviour: output modes, exit codes and error reporting."""

from __future__ import annotations

import json

import pytest

from safetensors_print.render import RULE_WIDTH
from safetensors_print.cli import (
    EXIT_OK,
    EXIT_SPECIFICATION_VIOLATIONS,
    EXIT_UNREADABLE,
    EXIT_USAGE,
    main,
)

from .support import build_file_bytes, float32_bytes, write_file

VALID_HEADER = {
    "__metadata__": {"format": "pt", "config": '{"layers": 2}'},
    "weight": {"dtype": "F32", "shape": [2, 1], "data_offsets": [0, 8]},
}


@pytest.fixture
def valid_file(tmp_path):
    return write_file(
        tmp_path, "model.safetensors", build_file_bytes(VALID_HEADER, float32_bytes(1.5, -2.25))
    )


@pytest.fixture
def file_with_a_gap(tmp_path):
    header = {"a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}
    return write_file(tmp_path, "gap.safetensors", build_file_bytes(header, bytes(8)))


def test_default_run_succeeds_and_prints_every_section(valid_file, capsys):
    assert main([valid_file]) == EXIT_OK

    output = capsys.readouterr().out
    for heading in (
        "FILE",
        "INTEGRITY",
        "ISSUES",
        "__METADATA__",
        "TENSORS",
        "DTYPE SUMMARY",
        "HEADER JSON",
    ):
        assert heading in output


def test_default_run_reports_tensor_details(valid_file, capsys):
    main([valid_file])

    output = capsys.readouterr().out
    assert "weight" in output
    assert "F32" in output
    assert "2x1" in output
    assert "IEEE 754 single precision" in output


def test_default_run_embeds_the_header_with_sorted_keys(valid_file, capsys):
    main([valid_file])

    header_section = capsys.readouterr().out.split("HEADER JSON")[1]
    assert '"weight"' in header_section
    assert '"dtype": "F32"' in header_section
    assert header_section.index('"__metadata__"') < header_section.index('"weight"')


def test_header_section_sorts_keys(tmp_path, capsys):
    header = {
        "zebra": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]},
        "alpha": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(2)))

    main([path, "--header"])

    output = capsys.readouterr().out
    assert output.index('"alpha"') < output.index('"zebra"')


def test_verbose_adds_decoded_values_and_hex_dumps(valid_file, capsys):
    assert main([valid_file, "--verbose"]) == EXIT_OK

    output = capsys.readouterr().out
    assert "first elements" in output
    assert "1.5" in output
    assert "-2.25" in output
    assert "first 8 bytes:" in output
    assert "absolute file offsets" in output


def test_metadata_is_rendered_as_json_with_encoded_values_decoded(valid_file, capsys):
    """A JSON-encoded metadata value is expanded in place, not printed as one escaped line."""
    main([valid_file])

    output = capsys.readouterr().out
    metadata_section = output.split("__METADATA__")[1].split("TENSORS")[0]
    assert '"format": "pt"' in metadata_section
    assert "shown decoded" in metadata_section
    assert '"layers": 2' in metadata_section
    assert '{\\"layers\\": 2}' not in metadata_section
    # The header JSON section reads the same way, rather than repeating the escaped line.
    assert "shown decoded" in output.split("HEADER JSON")[1]


def test_metadata_json_rendering_does_not_need_verbose(valid_file, capsys):
    main([valid_file])
    default_output = capsys.readouterr().out

    main([valid_file, "--verbose"])
    verbose_output = capsys.readouterr().out

    assert "shown decoded" in default_output
    assert "shown decoded" in verbose_output


def test_metadata_scalar_strings_are_not_turned_into_numbers(tmp_path, capsys):
    """A numeric-looking metadata value is a string in the file and must stay one."""
    header = {
        "__metadata__": {"training_step": "5000"},
        "w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, b"\x00"))

    main([path])

    assert '"training_step": "5000"' in capsys.readouterr().out


def test_no_metadata_line_wraps_beyond_the_rule_width(tmp_path, capsys):
    """Long JSON-encoded values used to print as a single unreadable line."""
    nested = {"key_number_{}".format(index): "value " * 12 for index in range(20)}
    header = {
        "__metadata__": {"architecture": json.dumps(nested)},
        "w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, b"\x00"))

    main([path])

    metadata_lines = capsys.readouterr().out.split("__METADATA__")[1].split("TENSORS")[0]
    assert max(len(line) for line in metadata_lines.splitlines()) <= RULE_WIDTH


def test_specification_violation_sets_exit_code_one_but_still_prints(file_with_a_gap, capsys):
    assert main([file_with_a_gap]) == EXIT_SPECIFICATION_VIOLATIONS

    output = capsys.readouterr().out
    assert "claimed by no tensor" in output
    assert "-- unclaimed gap --" in output
    assert "HEADER JSON" in output


def test_unsorted_offsets_alone_do_not_fail_the_run(tmp_path, capsys):
    header = {
        "b": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]},
        "a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(2)))

    assert main([path]) == EXIT_OK
    assert "WARNING" in capsys.readouterr().out


def test_unknown_dtype_is_counted_separately_from_agreement(tmp_path, capsys):
    """An unknown dtype makes the expected size unknowable, which must not read as agreement."""
    header = {
        "known": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
        "mystery": {"dtype": "F128", "shape": [1], "data_offsets": [1, 17]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(17)))

    assert main([path]) == EXIT_SPECIFICATION_VIOLATIONS

    output = capsys.readouterr().out
    assert "1 agree, 1 undeterminable" in output
    assert "F128 (UNKNOWN)" in output


def test_missing_file_reports_an_error_on_stderr(tmp_path, capsys):
    assert main([str(tmp_path / "absent.safetensors")]) == EXIT_UNREADABLE

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "safetensors-print:" in captured.err


def test_unparsable_header_reports_an_error_on_stderr(tmp_path, capsys):
    path = write_file(tmp_path, "truncated.safetensors", b"\x01\x02")

    assert main([path]) == EXIT_UNREADABLE
    assert "too short" in capsys.readouterr().err


@pytest.mark.parametrize("withdrawn", ["--json-only", "--pretty"])
def test_withdrawn_options_are_rejected(valid_file, capsys, withdrawn):
    with pytest.raises(SystemExit) as raised:
        main([valid_file, withdrawn])

    assert raised.value.code == EXIT_USAGE


@pytest.mark.parametrize("abbreviation", ["--met", "--metadata-r", "--verb", "--so", "--tens"])
def test_abbreviated_options_are_rejected(valid_file, capsys, abbreviation):
    """A prefix that works today would break the day a longer option makes it ambiguous."""
    with pytest.raises(SystemExit) as raised:
        main([valid_file, abbreviation])

    assert raised.value.code == EXIT_USAGE
    assert "unrecognized arguments" in capsys.readouterr().err


def test_missing_filename_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as raised:
        main([])

    assert raised.value.code == EXIT_USAGE


def test_help_lists_the_documented_options(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["--help"])

    assert raised.value.code == EXIT_OK
    output = capsys.readouterr().out
    for option in ("--verbose", "--summary", "--issues", "--tensors", "--header", "--metadata-raw"):
        assert option in output
    assert "exit codes:" in output


def test_version_is_reported(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["--version"])

    assert raised.value.code == EXIT_OK
    assert "safetensors-print" in capsys.readouterr().out


def test_tensors_are_listed_once_in_offset_order_by_default(tmp_path, capsys):
    """The tensor table doubles as the data buffer map, so nothing is listed twice."""
    header = {
        "zebra": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
        "alpha": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(2)))

    assert main([path]) == EXIT_OK

    output = capsys.readouterr().out
    table = output.split("TENSORS")[1].split("DTYPE SUMMARY")[0]
    assert table.count("zebra") == 1
    assert table.count("alpha") == 1
    assert table.index("zebra") < table.index("alpha")
    assert "ordered by offset" in output


def test_sort_by_name_reorders_the_same_single_table(tmp_path, capsys):
    header = {
        "zebra": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
        "alpha": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(2)))

    assert main([path, "--sort", "name"]) == EXIT_OK

    output = capsys.readouterr().out
    table = output.split("TENSORS")[1].split("DTYPE SUMMARY")[0]
    assert table.count("zebra") == 1
    assert table.index("alpha") < table.index("zebra")
    assert "ordered by name" in output


def test_offsets_are_two_columns_rather_than_a_joined_range(tmp_path, capsys):
    header = {
        "w": {"dtype": "U8", "shape": [3], "data_offsets": [0, 3]},
        "x": {"dtype": "U8", "shape": [1], "data_offsets": [3, 4]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(4)))

    main([path])

    table = capsys.readouterr().out.split("TENSORS")[1].split("DTYPE SUMMARY")[0]
    heading = next(line for line in table.splitlines() if "OFFSET_BEGIN" in line)
    assert heading.index("OFFSET_BEGIN") < heading.index("OFFSET_END")
    assert "DATA_OFFSETS" not in table
    assert ".." not in table


def test_a_table_with_no_rows_says_so_rather_than_printing_a_bare_heading(tmp_path, capsys):
    """Ordering by name drops gaps, so a file of only unusable entries tabulates nothing."""
    header = {"broken": {"dtype": "F32", "shape": [-2], "data_offsets": [0, 8]}}
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(8)))

    assert main([path, "--tensors", "--sort", "name"]) == EXIT_SPECIFICATION_VIOLATIONS

    output = capsys.readouterr().out
    assert "The header declares no tensors." in output
    assert section_titles(output) == ["TENSORS"]


def test_a_gap_row_fills_both_offset_columns(file_with_a_gap, capsys):
    main([file_with_a_gap])

    table = capsys.readouterr().out.split("TENSORS")[1].split("DTYPE SUMMARY")[0]
    gap_row = next(line for line in table.splitlines() if "unclaimed gap" in line)
    assert gap_row.split()[-2:] == ["1", "8"]


def test_gaps_appear_in_place_only_when_ordered_by_offset(file_with_a_gap, capsys):
    main([file_with_a_gap])
    offset_table = capsys.readouterr().out.split("TENSORS")[1].split("DTYPE SUMMARY")[0]

    main([file_with_a_gap, "--sort", "name"])
    name_table = capsys.readouterr().out.split("TENSORS")[1].split("DTYPE SUMMARY")[0]

    assert "-- unclaimed gap --" in offset_table
    assert "-- unclaimed gap --" not in name_table


def test_invalid_sort_order_is_a_usage_error(valid_file, capsys):
    with pytest.raises(SystemExit) as raised:
        main([valid_file, "--sort", "sideways"])

    assert raised.value.code == EXIT_USAGE


def test_verbose_detail_section_is_absent_by_default(valid_file, capsys):
    main([valid_file])
    assert "TENSOR DETAIL" not in capsys.readouterr().out

    main([valid_file, "--verbose"])
    assert "TENSOR DETAIL" in capsys.readouterr().out


def test_metadata_flag_prints_only_the_metadata_object(valid_file, capsys):
    assert main([valid_file, "--metadata"]) == EXIT_OK

    output = capsys.readouterr().out
    assert "INTEGRITY" not in output
    assert "weight" not in output


def test_metadata_expands_encoded_values_by_default(valid_file, capsys):
    main([valid_file, "--metadata"])

    assert json.loads(capsys.readouterr().out) == {"format": "pt", "config": {"layers": 2}}


def test_metadata_output_is_valid_json_and_sorted(valid_file, capsys):
    """Expansion must not cost the output its parsability, so no annotating comments."""
    main([valid_file, "--metadata"])

    output = capsys.readouterr().out
    json.loads(output)
    assert "/*" not in output
    assert "shown decoded" not in output
    assert output.index('"config"') < output.index('"format"')


def test_metadata_raw_is_byte_faithful(valid_file, capsys):
    assert main([valid_file, "--metadata-raw"]) == EXIT_OK

    output = capsys.readouterr().out
    assert output.strip() == json.dumps(VALID_HEADER["__metadata__"], indent=2, sort_keys=True)
    assert json.loads(output)["config"] == '{"layers": 2}'


def test_metadata_leaves_values_that_are_not_encoded_json_alone(tmp_path, capsys):
    header = {
        "__metadata__": {"training_step": "5000", "notes": "autosave @ step 5000"},
        "w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, b"\x00"))

    assert main([path, "--metadata"]) == EXIT_OK

    assert json.loads(capsys.readouterr().out) == header["__metadata__"]


@pytest.mark.parametrize("form", ["--metadata", "--metadata-raw"])
def test_either_metadata_form_on_a_file_without_metadata_prints_an_empty_object(
    tmp_path, capsys, form
):
    header = {"w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, b"\x00"))

    assert main([path, form]) == EXIT_OK

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {}
    assert "declares no __metadata__ key" in captured.err


def test_the_two_metadata_forms_are_mutually_exclusive(valid_file, capsys):
    with pytest.raises(SystemExit) as raised:
        main([valid_file, "--metadata", "--metadata-raw"])

    assert raised.value.code == EXIT_USAGE
    assert "not allowed with" in capsys.readouterr().err


@pytest.mark.parametrize("form", ["--metadata", "--metadata-raw"])
@pytest.mark.parametrize("conflicting", ["--verbose", "--tensors", "--summary", "--header"])
def test_metadata_cannot_be_combined_with_sections(valid_file, capsys, form, conflicting):
    with pytest.raises(SystemExit) as raised:
        main([valid_file, form, conflicting])

    assert raised.value.code == EXIT_USAGE
    assert "cannot be combined with" in capsys.readouterr().err


def test_metadata_is_rejected_before_the_file_is_read(tmp_path, capsys):
    """A usage error must not depend on the file existing."""
    with pytest.raises(SystemExit) as raised:
        main([str(tmp_path / "absent.safetensors"), "--metadata", "--tensors"])

    assert raised.value.code == EXIT_USAGE


def test_the_annotating_comment_belongs_to_the_dump_only(valid_file, capsys):
    """The dump is prose and can say how a value is stored; the JSON output cannot."""
    main([valid_file])
    assert "shown decoded" in capsys.readouterr().out

    main([valid_file, "--metadata"])
    assert "shown decoded" not in capsys.readouterr().out


def section_titles(output):
    """The title of each section printed, read from between its two rules.

    Titles are compared rather than substrings of the whole dump, since a table can
    hold a column named after another section.
    """
    rule = "=" * RULE_WIDTH
    lines = output.splitlines()
    return [
        line.split(" (")[0]
        for index, line in enumerate(lines)
        if 0 < index < len(lines) - 1 and lines[index - 1] == rule and lines[index + 1] == rule
    ]


SECTION_TITLES = {
    "--summary": ["FILE", "INTEGRITY", "DTYPE SUMMARY"],
    "--issues": ["ISSUES"],
    "--tensors": ["TENSORS"],
    "--header": ["HEADER JSON"],
}


@pytest.mark.parametrize("flag", sorted(SECTION_TITLES))
def test_a_section_flag_prints_that_section_and_no_other(valid_file, capsys, flag):
    assert main([valid_file, flag]) == EXIT_OK

    assert section_titles(capsys.readouterr().out) == SECTION_TITLES[flag]


def test_section_flags_combine(valid_file, capsys):
    assert main([valid_file, "--summary", "--tensors"]) == EXIT_OK

    assert section_titles(capsys.readouterr().out) == ["FILE", "INTEGRITY", "TENSORS", "DTYPE SUMMARY"]


def test_combined_sections_keep_the_order_of_the_full_dump(valid_file, capsys):
    """Flag order must not reorder the output; the dump has one canonical order."""
    main([valid_file, "--header", "--issues", "--tensors", "--summary"])
    one_order = capsys.readouterr().out

    main([valid_file, "--summary", "--tensors", "--issues", "--header"])
    assert capsys.readouterr().out == one_order
    assert section_titles(one_order) == [
        "FILE",
        "INTEGRITY",
        "ISSUES",
        "TENSORS",
        "DTYPE SUMMARY",
        "HEADER JSON",
    ]


def test_no_section_flag_prints_every_section(valid_file, capsys):
    main([valid_file])

    assert section_titles(capsys.readouterr().out) == [
        "FILE",
        "INTEGRITY",
        "ISSUES",
        "__METADATA__",
        "TENSORS",
        "DTYPE SUMMARY",
        "HEADER JSON",
    ]


def test_verbose_narrows_to_the_selected_sections(valid_file, capsys):
    assert main([valid_file, "--tensors", "--verbose"]) == EXIT_OK

    output = capsys.readouterr().out
    assert section_titles(output) == ["TENSORS", "TENSOR DETAIL"]
    assert "first elements" in output


def test_verbose_is_rejected_when_the_tensors_are_not_selected(valid_file, capsys):
    with pytest.raises(SystemExit) as raised:
        main([valid_file, "--summary", "--verbose"])

    assert raised.value.code == EXIT_USAGE
    assert "--verbose adds detail to the tensors output" in capsys.readouterr().err


def test_sort_is_rejected_when_the_tensors_are_not_selected(valid_file, capsys):
    with pytest.raises(SystemExit) as raised:
        main([valid_file, "--summary", "--sort", "name"])

    assert raised.value.code == EXIT_USAGE
    assert "--sort orders the tensors table" in capsys.readouterr().err


def test_sort_is_accepted_with_the_tensors_and_with_no_selection(valid_file, capsys):
    assert main([valid_file, "--tensors", "--sort", "name"]) == EXIT_OK
    capsys.readouterr()
    assert main([valid_file, "--sort", "name"]) == EXIT_OK
    assert "ordered by name" in capsys.readouterr().out


def test_sort_is_rejected_alongside_the_metadata_forms(valid_file, capsys):
    with pytest.raises(SystemExit) as raised:
        main([valid_file, "--metadata", "--sort", "name"])

    assert raised.value.code == EXIT_USAGE
    assert "cannot be combined with --sort" in capsys.readouterr().err


def test_sort_applies_to_a_lone_tensors_section(file_with_a_gap, capsys):
    main([file_with_a_gap, "--tensors", "--sort", "name"])
    assert "-- unclaimed gap --" not in capsys.readouterr().out

    main([file_with_a_gap, "--tensors", "--sort", "offset"])
    assert "-- unclaimed gap --" in capsys.readouterr().out


def test_a_section_flag_still_reports_violations_in_its_exit_code(file_with_a_gap, capsys):
    assert main([file_with_a_gap, "--summary"]) == EXIT_SPECIFICATION_VIOLATIONS


def test_metadata_reports_violations_in_its_exit_code(file_with_a_gap, capsys):
    assert main([file_with_a_gap, "--metadata"]) == EXIT_SPECIFICATION_VIOLATIONS


def test_metadata_flag_reports_specification_violations_in_its_exit_code(tmp_path, capsys):
    header = {
        "__metadata__": {"note": "hi"},
        "a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(8)))

    assert main([path, "--metadata"]) == EXIT_SPECIFICATION_VIOLATIONS
    assert json.loads(capsys.readouterr().out) == {"note": "hi"}
