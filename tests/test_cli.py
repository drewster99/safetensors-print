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


def test_only_json_only_prints_the_header_verbatim(valid_file, capsys):
    """The dump expands JSON-encoded values for reading; --json-only stays byte-faithful."""
    main([valid_file])
    dump = capsys.readouterr().out

    main([valid_file, "--json-only"])
    verbatim = capsys.readouterr().out

    assert verbatim.strip() == json.dumps(VALID_HEADER, indent=2, sort_keys=True)
    assert '{\\"layers\\": 2}' in verbatim
    assert '{\\"layers\\": 2}' not in dump


def test_json_only_prints_nothing_but_the_header(valid_file, capsys):
    assert main([valid_file, "--json-only"]) == EXIT_OK

    output = capsys.readouterr().out
    assert json.loads(output) == VALID_HEADER
    assert "INTEGRITY" not in output


def test_json_only_sorts_keys(tmp_path, capsys):
    header = {
        "zebra": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]},
        "alpha": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(2)))

    main([path, "--json-only"])

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


def test_verbose_and_json_only_are_mutually_exclusive(valid_file, capsys):
    with pytest.raises(SystemExit) as raised:
        main([valid_file, "--verbose", "--json-only"])

    assert raised.value.code == EXIT_USAGE
    assert "not allowed with" in capsys.readouterr().err


def test_missing_filename_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as raised:
        main([])

    assert raised.value.code == EXIT_USAGE


def test_help_lists_the_documented_options(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["--help"])

    assert raised.value.code == EXIT_OK
    output = capsys.readouterr().out
    assert "--verbose" in output
    assert "--json-only" in output
    assert "--pretty" in output
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
    assert json.loads(output) == VALID_HEADER["__metadata__"]
    assert "INTEGRITY" not in output
    assert "weight" not in output


def test_metadata_flag_output_is_verbatim_and_sorted(valid_file, capsys):
    """It must stay strictly valid JSON so it can be piped onward."""
    main([valid_file, "--metadata"])

    output = capsys.readouterr().out
    assert output.strip() == json.dumps(VALID_HEADER["__metadata__"], indent=2, sort_keys=True)
    assert output.index('"config"') < output.index('"format"')
    assert "/*" not in output


def test_metadata_flag_on_a_file_without_metadata_prints_an_empty_object(tmp_path, capsys):
    header = {"w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, b"\x00"))

    assert main([path, "--metadata"]) == EXIT_OK

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {}
    assert "declares no __metadata__ key" in captured.err


def test_metadata_flag_is_mutually_exclusive_with_the_other_output_modes(valid_file, capsys):
    for conflicting in ("--verbose", "--json-only"):
        with pytest.raises(SystemExit) as raised:
            main([valid_file, "--metadata", conflicting])
        assert raised.value.code == EXIT_USAGE


def test_pretty_expands_encoded_metadata_values(valid_file, capsys):
    assert main([valid_file, "--metadata", "--pretty"]) == EXIT_OK

    output = capsys.readouterr().out
    assert json.loads(output) == {"format": "pt", "config": {"layers": 2}}
    assert "INTEGRITY" not in output


def test_pretty_output_is_still_valid_json(valid_file, capsys):
    """Expansion must not cost the output its parsability, so no annotating comments."""
    for mode in ("--metadata", "--json-only"):
        main([valid_file, mode, "--pretty"])

        output = capsys.readouterr().out
        json.loads(output)
        assert "/*" not in output
        assert "shown decoded" not in output


def test_pretty_leaves_values_that_are_not_encoded_json_alone(tmp_path, capsys):
    header = {
        "__metadata__": {"training_step": "5000", "notes": "autosave @ step 5000"},
        "w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, b"\x00"))

    assert main([path, "--metadata", "--pretty"]) == EXIT_OK

    output = capsys.readouterr().out
    assert json.loads(output) == header["__metadata__"]


def test_pretty_expands_encoded_values_under_json_only_too(valid_file, capsys):
    assert main([valid_file, "--json-only", "--pretty"]) == EXIT_OK

    parsed = json.loads(capsys.readouterr().out)
    assert parsed["__metadata__"]["config"] == {"layers": 2}
    assert parsed["weight"] == VALID_HEADER["weight"]


def test_pretty_without_metadata_or_json_only_is_a_usage_error(valid_file, capsys):
    with pytest.raises(SystemExit) as raised:
        main([valid_file, "--pretty"])

    assert raised.value.code == EXIT_USAGE
    assert "--pretty applies to" in capsys.readouterr().err


def test_pretty_is_rejected_before_the_file_is_read(tmp_path, capsys):
    """The usage error must not depend on the file existing."""
    with pytest.raises(SystemExit) as raised:
        main([str(tmp_path / "absent.safetensors"), "--pretty"])

    assert raised.value.code == EXIT_USAGE


def test_pretty_metadata_still_reports_violations_in_its_exit_code(file_with_a_gap, capsys):
    assert main([file_with_a_gap, "--metadata", "--pretty"]) == EXIT_SPECIFICATION_VIOLATIONS


def test_pretty_on_a_file_without_metadata_prints_an_empty_object(tmp_path, capsys):
    header = {"w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, b"\x00"))

    assert main([path, "--metadata", "--pretty"]) == EXIT_OK

    captured = capsys.readouterr()
    assert captured.out.strip() == "{}"
    assert "declares no __metadata__ key" in captured.err


def test_metadata_without_pretty_stays_byte_faithful(valid_file, capsys):
    """--pretty is the only thing that turns an encoded string into structure."""
    main([valid_file, "--metadata"])

    assert json.loads(capsys.readouterr().out)["config"] == '{"layers": 2}'


def test_the_annotating_comment_belongs_to_the_dump_only(valid_file, capsys):
    """The dump is prose and can say how a value is stored; the JSON modes cannot."""
    main([valid_file])
    assert "shown decoded" in capsys.readouterr().out

    main([valid_file, "--metadata", "--pretty"])
    assert "shown decoded" not in capsys.readouterr().out


def test_metadata_flag_reports_specification_violations_in_its_exit_code(tmp_path, capsys):
    header = {
        "__metadata__": {"note": "hi"},
        "a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(8)))

    assert main([path, "--metadata"]) == EXIT_SPECIFICATION_VIOLATIONS
    assert json.loads(capsys.readouterr().out) == {"note": "hi"}
