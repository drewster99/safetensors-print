"""Command line entry point for safetensors-print."""

from __future__ import annotations

import argparse
import os
import sys
from typing import FrozenSet, Iterable, List, Optional, TextIO

from . import __version__
from .reader import SafetensorsFormatError, read_report
from .render import (
    ALL_SECTIONS,
    DEFAULT_SECTIONS,
    SORT_BY_OFFSET,
    SORT_ORDERS,
    Section,
    expanded_json_text,
    nothing_to_show_explanation,
    pretty_header_json,
    render_report,
)

EXIT_OK = 0
EXIT_SPECIFICATION_VIOLATIONS = 1
EXIT_USAGE = 2
EXIT_UNREADABLE = 3

# The command line flag that selects each section of the dump.
_SECTION_FLAGS = {
    "summary": Section.SUMMARY,
    "issues": Section.ISSUES,
    "tensors": Section.TENSORS,
    "header": Section.HEADER,
}

_DESCRIPTION = """\
Print what a .safetensors file states about itself: the byte layout, the
__metadata__ block, every tensor's dtype, shape and size, a map of the data
buffer accounting for every byte, and the header JSON pretty-printed with
sorted keys.

With no section selected it prints --summary --issues: how the file is laid
out, whether it holds together, and what is wrong with it if anything.
--tensors and --header add the rest, --all prints every section, and they
combine.

--metadata is the exception: it prints the __metadata__ block on its own as
JSON a pipeline can consume, expanding values that themselves hold JSON.
--metadata-raw prints the same block exactly as the file stores it.
"""

_EPILOG = """\
exit codes:
  0  the file was printed and conforms to the safetensors specification
  1  the file was printed but violates the specification
  2  the command line was invalid
  3  the file could not be read or its header could not be parsed
"""


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safetensors-print",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # An unambiguous prefix such as --meta would be accepted by default, becoming an
        # undocumented spelling that a later option could turn ambiguous and break.
        allow_abbrev=False,
    )
    parser.add_argument("filename", help="path to the .safetensors file to inspect")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print the FILE, INTEGRITY and DTYPE SUMMARY sections: the layout, what the "
        "header claims and whether it holds together, and the per-dtype totals",
    )
    parser.add_argument(
        "--issues",
        action="store_true",
        help="print the ISSUES section: every departure from the safetensors specification",
    )
    parser.add_argument(
        "--tensors",
        action="store_true",
        help="print the TENSORS table: one row per tensor, plus any unclaimed gap when "
        "ordered by offset",
    )
    parser.add_argument(
        "--header",
        action="store_true",
        help="print the HEADER JSON section: the whole header pretty-printed with sorted "
        "keys, with JSON-encoded values expanded and annotated",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="print every section, including __METADATA__, which no flag of its own "
        "selects. Cannot be combined with the individual section flags",
    )

    metadata_form = parser.add_mutually_exclusive_group()
    metadata_form.add_argument(
        "--metadata",
        action="store_true",
        help="print only the __metadata__ object, as JSON with sorted keys, expanding every "
        "value that itself holds a JSON object or array. Still valid JSON, so it pipes, but "
        "no longer byte-faithful. Prints {} when the file declares no metadata, and says so "
        "on stderr",
    )
    metadata_form.add_argument(
        "--metadata-raw",
        action="store_true",
        help="as --metadata, but byte-faithful: a value the file stores as a JSON-encoded "
        "string is printed as that string, escaping and all",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="add a TENSOR DETAIL section to the tensors output: decoded leading element "
        "values, hex dumps of the head and tail of every data segment, and absolute file "
        "offsets",
    )
    parser.add_argument(
        "--sort",
        choices=SORT_ORDERS,
        # Left unset rather than defaulted, so that naming it where it can have no effect
        # is distinguishable from not naming it at all.
        default=None,
        help="order of the TENSORS table: 'offset' (default) lays out the data buffer and "
        "shows any unclaimed gaps in place; 'name' sorts alphabetically",
    )
    parser.add_argument("--version", action="version", version="%(prog)s {}".format(__version__))
    return parser


def _selected_sections(arguments: argparse.Namespace) -> FrozenSet[Section]:
    """The sections to print: every one, those named, or the default pair."""
    if arguments.all:
        return ALL_SECTIONS
    selected = frozenset(
        section for flag, section in _SECTION_FLAGS.items() if getattr(arguments, flag)
    )
    return selected if selected else DEFAULT_SECTIONS


def _reject_unusable_combinations(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    """Refuse flag combinations under which a flag would do nothing.

    Accepting them silently would leave the user believing the ignored flag had taken
    effect, which is worse than being told the combination is not one we serve.
    """
    named_sections = ["--" + flag for flag in _SECTION_FLAGS if getattr(arguments, flag)]
    tensor_options = (["--verbose"] if arguments.verbose else []) + (
        ["--sort"] if arguments.sort is not None else []
    )

    if arguments.metadata or arguments.metadata_raw:
        conflicting = named_sections + (["--all"] if arguments.all else []) + tensor_options
        if conflicting:
            parser.error(
                "{} prints the metadata on its own and cannot be combined with {}".format(
                    "--metadata-raw" if arguments.metadata_raw else "--metadata",
                    ", ".join(conflicting),
                )
            )
        return

    if arguments.all and named_sections:
        parser.error(
            "--all already prints every section, so {} adds nothing".format(", ".join(named_sections))
        )

    if Section.TENSORS not in _selected_sections(arguments):
        if arguments.verbose:
            parser.error(
                "--verbose adds detail to the tensors output, which this run does not "
                "print; add --tensors or --all"
            )
        if arguments.sort is not None:
            parser.error(
                "--sort orders the tensors table, which this run does not print; "
                "add --tensors or --all"
            )


def _write_lines(lines: Iterable[str], stream: TextIO) -> None:
    """Write `lines`, treating a closed downstream pipe as a normal end of output."""
    try:
        for line in lines:
            stream.write(line + "\n")
        stream.flush()
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stream.fileno())
        raise SystemExit(EXIT_OK)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    _reject_unusable_combinations(parser, arguments)

    try:
        report = read_report(arguments.filename)
    except SafetensorsFormatError as error:
        print("safetensors-print: {}".format(error), file=sys.stderr)
        return EXIT_UNREADABLE
    except OSError as error:
        print(
            "safetensors-print: cannot read {}: {}".format(arguments.filename, error.strerror or error),
            file=sys.stderr,
        )
        return EXIT_UNREADABLE

    if arguments.metadata or arguments.metadata_raw:
        # An empty object on stdout is a fact about the file, not an absence of output,
        # so whichever reason produced it is said out loud, in the dump's own words.
        explanation = nothing_to_show_explanation(report.metadata_declaration)
        if explanation is not None:
            print("safetensors-print: {}: {}".format(arguments.filename, explanation), file=sys.stderr)
        render_metadata = pretty_header_json if arguments.metadata_raw else expanded_json_text
        _write_lines([render_metadata(report.metadata)], sys.stdout)
    else:
        _write_lines(
            render_report(
                report,
                sections=_selected_sections(arguments),
                verbose=arguments.verbose,
                sort_by=arguments.sort or SORT_BY_OFFSET,
            ),
            sys.stdout,
        )

    return EXIT_SPECIFICATION_VIOLATIONS if report.has_errors else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
