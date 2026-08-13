#!/usr/bin/env python3
"""Run every command line combination against every file given, and check the results.

    python3 scripts/run-option-matrix.py tests/corpus [more paths ...]

Each file is run through all 96 combinations of the four section flags, --verbose and
--sort, plus the two metadata forms and the combinations that are meant to be refused.
Directories are searched for .safetensors files.

What each run is checked against is worked out here, independently of the tool: which
combinations are usable, which sections each one should print, and what the metadata
should come out as, read from the file's own bytes. A disagreement is a failure whichever
side is wrong, which is the point of restating the rules rather than importing them.

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import struct
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

SECTION_FLAGS = ["--summary", "--issues", "--tensors", "--header"]
SORT_VALUES = [None, "offset", "name"]
METADATA_FORMS = ["--metadata", "--metadata-raw"]

EXIT_OK = 0
EXIT_SPECIFICATION_VIOLATIONS = 1
EXIT_USAGE = 2
EXIT_UNREADABLE = 3

RULE = "=" * 100


class Case:
    """One invocation and what it is expected to produce."""

    def __init__(self, arguments: Sequence[str], description: str):
        self.arguments = list(arguments)
        self.description = description

    def __repr__(self) -> str:
        return " ".join(self.arguments)


def powerset(items: Sequence[str]):
    for size in range(len(items) + 1):
        for combination in itertools.combinations(items, size):
            yield list(combination)


def option_cases() -> List[Case]:
    """Every combination of the options that take a file, valid or not."""
    cases = []
    for sections in powerset(SECTION_FLAGS):
        for verbose in (False, True):
            for sort in SORT_VALUES:
                arguments = list(sections)
                if verbose:
                    arguments.append("--verbose")
                if sort is not None:
                    arguments += ["--sort", sort]
                cases.append(Case(arguments, "sections"))

    for form in METADATA_FORMS:
        cases.append(Case([form], "metadata"))
        for conflicting in SECTION_FLAGS + ["--verbose"]:
            cases.append(Case([form, conflicting], "metadata conflict"))
        cases.append(Case([form, "--sort", "name"], "metadata conflict"))
    cases.append(Case(METADATA_FORMS, "metadata conflict"))
    return cases


def is_usage_error(arguments: Sequence[str]) -> bool:
    """The rules for refusing a combination, restated independently of the tool."""
    sections = [flag for flag in SECTION_FLAGS if flag in arguments]
    verbose = "--verbose" in arguments
    sorted_ = "--sort" in arguments
    metadata_forms = [form for form in METADATA_FORMS if form in arguments]

    if len(metadata_forms) > 1:
        return True
    if metadata_forms:
        return bool(sections) or verbose or sorted_
    if sections and "--tensors" not in sections:
        return verbose or sorted_
    return False


def expected_section_titles(arguments: Sequence[str]) -> List[str]:
    """The titles the dump should print, in the order the full dump uses."""
    selected = [flag for flag in SECTION_FLAGS if flag in arguments]
    chosen = set(selected) if selected else set(SECTION_FLAGS)
    everything = not selected

    titles = []
    if "--summary" in chosen:
        titles += ["FILE", "INTEGRITY"]
    if "--issues" in chosen:
        titles += ["ISSUES"]
    if everything:
        titles += ["__METADATA__"]
    if "--tensors" in chosen:
        titles += ["TENSORS"]
    if "--summary" in chosen:
        titles += ["DTYPE SUMMARY"]
    if "--verbose" in arguments and "--tensors" in chosen:
        titles += ["TENSOR DETAIL"]
    if "--header" in chosen:
        titles += ["HEADER JSON"]
    return titles


def printed_section_titles(output: str) -> List[str]:
    lines = output.splitlines()
    return [
        line.split(" (")[0]
        for index, line in enumerate(lines)
        if 0 < index < len(lines) - 1 and lines[index - 1] == RULE and lines[index + 1] == RULE
    ]


# The reference implementation's ceiling, which the tool refuses to read past.
MAXIMUM_HEADER_SIZE = 100_000_000


def read_header_independently(path: str) -> Optional[Dict[str, Any]]:
    """The file's header, parsed here rather than by the tool. None if unreadable.

    Uses `raw_decode` so that trailing padding, which the format allows, does not read
    as damage: the tool judges the padding's contents separately.
    """
    try:
        with open(path, "rb") as handle:
            length_field = handle.read(8)
            if len(length_field) < 8:
                return None
            declared = struct.unpack("<Q", length_field)[0]
            if declared > MAXIMUM_HEADER_SIZE:
                return None
            raw = handle.read(declared)
            if len(raw) < declared:
                return None
            header, _ = json.JSONDecoder().raw_decode(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return header if isinstance(header, dict) else None


def expected_metadata(header: Dict[str, Any], expand: bool) -> Dict[str, Any]:
    """What the metadata forms should print, derived from the header read above."""
    raw = header.get("__metadata__")
    if not isinstance(raw, dict):
        return {}

    metadata: Dict[str, Any] = {}
    for key, value in raw.items():
        as_string = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        if expand:
            try:
                decoded = json.loads(as_string)
            except ValueError:
                decoded = None
            if isinstance(decoded, (dict, list)):
                metadata[key] = decoded
                continue
        metadata[key] = as_string
    return metadata


def run(command: Sequence[str], arguments: Sequence[str]) -> Tuple[int, str, str]:
    completed = subprocess.run(
        list(command) + list(arguments),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.returncode, completed.stdout, completed.stderr


def display_path(path: str) -> str:
    relative = os.path.relpath(path)
    return path if relative.startswith("..") else relative


def check_file(command: Sequence[str], path: str, cases: Sequence[Case]) -> Tuple[List[str], int]:
    """Run every case against one file, returning its failures and how many runs it took."""
    failures: List[str] = []
    header = read_header_independently(path)
    readable = header is not None

    baseline_code, baseline_output, _ = run(command, [path])
    if readable and baseline_code not in (EXIT_OK, EXIT_SPECIFICATION_VIOLATIONS):
        failures.append("{}: readable file exited {}".format(path, baseline_code))
    if not readable and baseline_code != EXIT_UNREADABLE:
        failures.append("{}: unreadable file exited {}, wanted 3".format(path, baseline_code))

    for case in cases:
        arguments = [path] + case.arguments
        code, output, errors = run(command, arguments)
        label = "{} {}".format(os.path.basename(path), " ".join(case.arguments) or "(no options)")

        if "Traceback (most recent call last)" in errors:
            failures.append("{}: traceback\n{}".format(label, errors.strip()[:400]))
            continue

        if is_usage_error(case.arguments):
            if code != EXIT_USAGE:
                failures.append("{}: expected usage error, exited {}".format(label, code))
            elif output:
                failures.append("{}: usage error still wrote to stdout".format(label))
            elif "error:" not in errors:
                failures.append("{}: usage error said nothing on stderr".format(label))
            continue

        if not readable:
            if code != EXIT_UNREADABLE:
                failures.append("{}: expected exit 3, got {}".format(label, code))
            elif output:
                failures.append("{}: unreadable file still wrote to stdout".format(label))
            elif "safetensors-print:" not in errors:
                failures.append("{}: unreadable file explained nothing".format(label))
            continue

        if code != baseline_code:
            failures.append(
                "{}: exited {}, but the file itself exits {}".format(label, code, baseline_code)
            )

        metadata_form = next((form for form in METADATA_FORMS if form in case.arguments), None)
        if metadata_form is not None:
            try:
                parsed = json.loads(output)
            except ValueError as error:
                failures.append("{}: output is not valid JSON ({})".format(label, error))
                continue
            wanted = expected_metadata(header, expand=metadata_form == "--metadata")
            if parsed != wanted:
                failures.append("{}: metadata differs from the file's own".format(label))
            if "/*" in output:
                failures.append("{}: JSON output carries a comment".format(label))
            continue

        titles = printed_section_titles(output)
        wanted_titles = expected_section_titles(case.arguments)
        if titles != wanted_titles:
            failures.append("{}: printed {}, wanted {}".format(label, titles, wanted_titles))
        if not output.strip():
            failures.append("{}: printed nothing".format(label))

    # The same invocation twice must produce the same bytes.
    repeated_code, repeated_output, _ = run(command, [path])
    if (repeated_code, repeated_output) != (baseline_code, baseline_output):
        failures.append("{}: two identical runs disagreed".format(path))

    return failures, len(cases) + 2


def check_command_line_errors(command: Sequence[str], sample: str) -> Tuple[List[str], int]:
    """Cases that do not depend on a file's contents."""
    failures = []
    expectations = [
        ([], EXIT_USAGE, "no arguments"),
        (["--help"], EXIT_OK, "--help"),
        (["--version"], EXIT_OK, "--version"),
        ([sample, "--json-only"], EXIT_USAGE, "a withdrawn option"),
        ([sample, "--pretty"], EXIT_USAGE, "a withdrawn option"),
        ([sample, "--json"], EXIT_USAGE, "an abbreviation"),
        ([sample, "--met"], EXIT_USAGE, "an abbreviation"),
        ([sample, "--metadata-r"], EXIT_USAGE, "an abbreviation"),
        ([sample, "--verb"], EXIT_USAGE, "an abbreviation"),
        ([sample, "--tens"], EXIT_USAGE, "an abbreviation"),
        ([sample, "--nonsense"], EXIT_USAGE, "an unknown option"),
        ([sample, "--sort"], EXIT_USAGE, "--sort without a value"),
        ([sample, "--tensors", "--sort", "sideways"], EXIT_USAGE, "an invalid sort order"),
        ([sample, sample], EXIT_USAGE, "two filenames"),
        (["/no/such/file.safetensors"], EXIT_UNREADABLE, "a missing file"),
        (["/etc"], EXIT_UNREADABLE, "a directory"),
    ]
    for arguments, wanted, description in expectations:
        code, output, errors = run(command, arguments)
        if code != wanted:
            failures.append("{} exited {}, wanted {}".format(description, code, wanted))
        if "Traceback (most recent call last)" in errors:
            failures.append("{} produced a traceback".format(description))
        if wanted == EXIT_USAGE and output:
            failures.append("{} wrote to stdout".format(description))

    # A closed pipe is a normal end of output, not a crash.
    piped = subprocess.run(
        "{} {} | head -2".format(
            " ".join(command), sample.replace(" ", "\\ ")
        ),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if "BrokenPipeError" in piped.stderr or "Traceback" in piped.stderr:
        failures.append("a closed pipe produced a traceback")
    return failures, len(expectations) + 1


def gather_files(paths: Sequence[str]) -> List[str]:
    files = []
    for path in paths:
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                files += [
                    os.path.join(root, name)
                    for name in sorted(names)
                    if name.endswith(".safetensors")
                ]
        elif os.path.exists(path):
            files.append(path)
        else:
            print("no such path: {}".format(path), file=sys.stderr)
    return sorted(set(files))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", help=".safetensors files, or directories of them")
    parser.add_argument(
        "--command",
        default="{} -m safetensors_print".format(sys.executable),
        help="the command under test (default: this interpreter's safetensors_print)",
    )
    parser.add_argument("--jobs", type=int, default=min(8, (os.cpu_count() or 2)))
    arguments = parser.parse_args(argv)

    command = arguments.command.split()
    files = gather_files(arguments.paths)
    if not files:
        print("no .safetensors files found", file=sys.stderr)
        return 2

    cases = option_cases()
    print(
        "{} files x {} option combinations, {} at a time\n".format(
            len(files), len(cases), arguments.jobs
        )
    )

    failures, runs = check_command_line_errors(command, files[0])
    print("command line checks: {}".format("{} failed".format(len(failures)) if failures else "passed"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.jobs) as pool:
        futures = {pool.submit(check_file, command, path, cases): path for path in files}
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            found, count = future.result()
            failures += found
            runs += count
            print(
                "{:<4} {:>12,} bytes  {}".format(
                    "FAIL" if found else "ok", os.path.getsize(path), display_path(path)
                )
            )

    print()
    if failures:
        print("{:,} of {:,} runs failed:".format(len(failures), runs))
        for failure in failures:
            print("  - {}".format(failure))
        return 1
    print("all {:,} runs behaved as expected".format(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
