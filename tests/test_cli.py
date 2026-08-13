"""Command line behaviour: output modes, exit codes and error reporting."""

from __future__ import annotations

import json

import pytest

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
        "DATA SEGMENT MAP",
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


def test_default_run_embeds_the_sorted_pretty_printed_header(valid_file, capsys):
    main([valid_file])

    output = capsys.readouterr().out
    assert json.dumps(VALID_HEADER, indent=2, sort_keys=True) in output


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


def test_verbose_expands_metadata_values_that_contain_json(valid_file, capsys):
    main([valid_file, "--verbose"])

    output = capsys.readouterr().out
    assert "decoded as JSON" in output
    assert '"layers": 2' in output


def test_default_run_does_not_expand_nested_metadata_json(valid_file, capsys):
    main([valid_file])

    assert "decoded as JSON" not in capsys.readouterr().out


def test_specification_violation_sets_exit_code_one_but_still_prints(file_with_a_gap, capsys):
    assert main([file_with_a_gap]) == EXIT_SPECIFICATION_VIOLATIONS

    output = capsys.readouterr().out
    assert "claimed by no tensor" in output
    assert "-- GAP --" in output
    assert "HEADER JSON" in output


def test_unsorted_offsets_alone_do_not_fail_the_run(tmp_path, capsys):
    header = {
        "b": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]},
        "a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
    }
    path = write_file(tmp_path, "m.safetensors", build_file_bytes(header, bytes(2)))

    assert main([path]) == EXIT_OK
    assert "WARNING" in capsys.readouterr().out


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
    assert "exit codes:" in output


def test_version_is_reported(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["--version"])

    assert raised.value.code == EXIT_OK
    assert "safetensors-print" in capsys.readouterr().out
