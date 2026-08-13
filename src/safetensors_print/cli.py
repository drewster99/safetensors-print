"""Command line entry point for safetensors-print."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List, Optional, TextIO

from . import __version__
from .reader import METADATA_KEY, SafetensorsFormatError, read_report
from .render import (
    SORT_BY_OFFSET,
    SORT_ORDERS,
    expanded_json_text,
    pretty_header_json,
    render_report,
)

EXIT_OK = 0
EXIT_SPECIFICATION_VIOLATIONS = 1
EXIT_USAGE = 2
EXIT_UNREADABLE = 3

_DESCRIPTION = """\
Print everything a .safetensors file states about itself: the byte layout, the
__metadata__ block, every tensor's dtype, shape and size, a map of the data
buffer accounting for every byte, and the header JSON pretty-printed with
sorted keys.

The dump expands metadata values that themselves hold JSON, so they read as
nested objects instead of one very long escaped line. --json-only and
--metadata print the file's verbatim JSON instead, for piping onward; add
--pretty to expand those values there too. --pretty output is still valid
JSON, but it no longer reproduces the file byte for byte.
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
    )
    parser.add_argument("filename", help="path to the .safetensors file to inspect")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--verbose",
        action="store_true",
        help="add a TENSOR DETAIL section: decoded leading element values, hex dumps of the "
        "head and tail of every data segment, and absolute file offsets",
    )
    output_mode.add_argument(
        "--json-only",
        action="store_true",
        help="print only the header JSON, pretty-printed with sorted keys, verbatim",
    )
    output_mode.add_argument(
        "--metadata",
        action="store_true",
        help="print only the __metadata__ object, pretty-printed with sorted keys, verbatim. "
        "Prints {} when the file declares no metadata, and says so on stderr",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="with --metadata or --json-only: expand every value that itself holds a JSON "
        "object or array, instead of printing it as a single escaped line. The output is "
        "still valid JSON and still pipes, but it no longer reproduces the file: a value "
        "stored as an encoded string comes out as the structure it holds. The default dump "
        "already expands these values",
    )
    parser.add_argument(
        "--sort",
        choices=SORT_ORDERS,
        default=SORT_BY_OFFSET,
        help="order of the TENSORS table: 'offset' (default) lays out the data buffer and "
        "shows any unclaimed gaps in place; 'name' sorts alphabetically. Ignored with "
        "--json-only and --metadata",
    )
    parser.add_argument("--version", action="version", version="%(prog)s {}".format(__version__))
    return parser


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

    # Silently ignoring --pretty on the default dump would suggest the dump had been
    # left unexpanded without it, which is the opposite of what happens.
    if arguments.pretty and not (arguments.metadata or arguments.json_only):
        parser.error(
            "--pretty applies to --metadata and --json-only; the default dump always "
            "expands JSON-encoded values"
        )

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

    render_json = expanded_json_text if arguments.pretty else pretty_header_json

    if arguments.metadata:
        if METADATA_KEY not in report.header:
            print(
                "safetensors-print: {} declares no {} key".format(arguments.filename, METADATA_KEY),
                file=sys.stderr,
            )
        _write_lines([render_json(report.metadata)], sys.stdout)
    elif arguments.json_only:
        _write_lines([render_json(report.header)], sys.stdout)
    else:
        _write_lines(
            render_report(report, verbose=arguments.verbose, sort_by=arguments.sort), sys.stdout
        )

    return EXIT_SPECIFICATION_VIOLATIONS if report.has_errors else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
