#!/usr/bin/env python3
"""Check our verdict on each file against the reference implementation's.

    pip install -e ".[reference]"
    python3 scripts/compare-with-reference.py tests/corpus

The two tools answer different questions. `safetensors` answers "can I load this?" and
refuses the whole file at the first fault. We answer "what does this file say about
itself?" and describe it whatever state it is in. So they disagree by design on damaged
files, and the table below is meant to be read rather than to be empty.

One invariant must hold all the same, and is asserted: if we exit 3, the reference must
refuse the file too. Refusing to describe a file the canonical reader reads happily
would mean we cannot read the format.

The other direction is reported but not asserted. Exiting 1 where the reference loads
the file is our judgement that something is wrong with a file that will nonetheless
load, which is sometimes exactly right -- but it is worth seeing the list and agreeing
with each entry, since the exit code is what a build script gates on.

Exits non-zero if the invariant is broken.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

try:
    from safetensors import safe_open
except ImportError:
    print(
        'the reference implementation is not installed: pip install -e ".[reference]"',
        file=sys.stderr,
    )
    raise SystemExit(2)

EXIT_OK = 0
EXIT_UNREADABLE = 3


def reference_verdict(path: str):
    """(loads, explanation) from the reference implementation."""
    try:
        with safe_open(path, framework="numpy") as handle:
            handle.keys()
            handle.metadata()
    except Exception as error:  # The reference raises a variety of types.
        return False, str(error).splitlines()[0]
    return True, ""


def gather_files(paths):
    files = []
    for path in paths:
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                files += [
                    os.path.join(root, name) for name in sorted(names) if name.endswith(".safetensors")
                ]
        else:
            files.append(path)
    return sorted(set(files))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--command", default="{} -m safetensors_print".format(sys.executable))
    arguments = parser.parse_args(argv)

    command = arguments.command.split()
    failures = []
    agreements = 0
    we_describe_it_refuses = []
    we_object_it_loads = []

    for path in gather_files(arguments.paths):
        loads, explanation = reference_verdict(path)
        code = subprocess.run(command + [path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        name = os.path.basename(path)

        if code == EXIT_UNREADABLE and loads:
            failures.append("{}: we called it unreadable, but the reference loads it".format(name))

        if loads == (code == EXIT_OK):
            agreements += 1
        elif loads:
            we_object_it_loads.append((name, code))
        else:
            we_describe_it_refuses.append((name, code, explanation))

    print("{} files agree".format(agreements))

    if we_describe_it_refuses:
        print(
            "\n{} we describe, the reference refuses (by design: we report rather than "
            "refuse):".format(len(we_describe_it_refuses))
        )
        for name, code, explanation in we_describe_it_refuses:
            print("  {:<36} exit {}  reference: {}".format(name, code, explanation[:56]))

    if we_object_it_loads:
        print(
            "\n{} we object to, the reference loads (check each of these is a judgement "
            "you want):".format(len(we_object_it_loads))
        )
        for name, code in we_object_it_loads:
            print("  {:<36} exit {}".format(name, code))

    if failures:
        print("\n{} invariant failure(s):".format(len(failures)))
        for failure in failures:
            print("  - {}".format(failure))
        return 1
    print("\nthe invariant holds: nothing we refuse is readable by the reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
