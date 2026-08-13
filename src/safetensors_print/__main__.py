"""Support `python -m safetensors_print`."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
