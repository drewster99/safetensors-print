# safetensors-print

Print everything a `.safetensors` file states about itself — the byte layout, the
`__metadata__` block, every tensor's dtype/shape/size, a map of the data buffer that
accounts for every byte, and the header JSON pretty-printed with sorted keys.

No third-party dependencies. It never loads tensor data into memory, so it opens a
100 GB checkpoint as quickly as a 100 KB one.

## Install

```sh
pip install safetensors-print
```

Or run it straight from a checkout:

```sh
python3 -m safetensors_print model.safetensors
```

## Usage

```
safetensors-print <filename.safetensors> [--summary] [--issues] [--tensors] [--header]
                  [--verbose] [--sort offset|name]
safetensors-print <filename.safetensors> (--metadata | --metadata-raw)
```

| Option | Effect |
| --- | --- |
| *(none)* | Full dump: every section below, plus `__METADATA__` |
| `--summary` | `FILE`, `INTEGRITY` and `DTYPE SUMMARY`: the layout, whether the header holds together, and the per-dtype totals |
| `--issues` | `ISSUES`: every departure from the specification |
| `--tensors` | `TENSORS`: one row per tensor, plus any unclaimed gap |
| `--header` | `HEADER JSON`: the whole header, keys sorted, encoded values expanded and annotated |
| `--verbose` | Adds `TENSOR DETAIL` to the tensors output: decoded leading element values, hex dumps of the head and tail of every segment, and absolute file offsets |
| `--sort offset\|name` | Order of the `TENSORS` table. `offset` (default) lays out the data buffer and shows unclaimed gaps in place; `name` sorts alphabetically |
| `--metadata` | Prints only `__metadata__`, as JSON, with encoded values expanded |
| `--metadata-raw` | Prints only `__metadata__`, as JSON, exactly as the file stores it |
| `--version` | Prints the version |
| `--help` | Usage |

Option abbreviations are not accepted: `--met` is an error, not a shorthand for
`--metadata`. A prefix that works today would break the day a longer option makes it
ambiguous.

### Selecting sections

The four section flags combine, and the output always follows the order of the full
dump regardless of the order they are given in:

```sh
safetensors-print model.safetensors --summary --issues
```

`--verbose` belongs to the tensors output, so it is rejected alongside a selection that
excludes `--tensors`, rather than being silently ignored.

### Reading the metadata

`--metadata` and `--metadata-raw` print the `__metadata__` block on its own, as JSON a
pipeline can consume. They cannot be combined with the section flags.

`--metadata` expands values that themselves hold JSON, so a configuration stored as an
encoded string can be queried as structure:

```sh
safetensors-print model.safetensors --metadata | jq .architecture.input_encoding
```

The output is still valid JSON; what it gives up is byte-faithfulness, since a value the
file holds as a string comes out as the object that string contains. It carries no
annotating comment for exactly that reason. `--metadata-raw` is the byte-faithful form,
reproducing what the file holds:

```sh
safetensors-print model.safetensors --metadata-raw | jq -r .architecture | jq .
```

Both print `{}` and note it on stderr when the file declares no metadata, so a pipeline
never receives empty input.

The `TENSORS` table is the only listing of tensors. Ordered by offset it doubles as the
map of the data buffer, with any unclaimed gaps shown in place, so no tensor is ever
printed twice.

The `__METADATA__` and `HEADER JSON` sections of the dump both print as JSON with keys
sorted. The specification defines `__metadata__` as a flat map of string to string, so a
model configuration has nowhere to go but into a JSON-encoded string; in the dump those
are expanded in place and annotated `/* stored as a JSON-encoded string, shown decoded
*/`, rather than printed as a single escaped line hundreds of characters wide.
Numeric-looking values such as `"5000"` are strings in the file and stay strings.

That annotation makes the dump readable but not machine-parsable, which is why
`--metadata` and `--metadata-raw` expand without annotating.

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Printed; the file conforms to the safetensors specification |
| 1 | Printed; the file violates the specification (see the `ISSUES` section) |
| 2 | The command line was invalid |
| 3 | The file could not be read, or its header could not be parsed |

Because violations exit non-zero while still printing, this works as a checkpoint
validator in a build script:

```sh
safetensors-print model.safetensors > /dev/null || echo "non-conforming"
```

## Example

```
====================================================================================================
FILE
====================================================================================================
  Path                            model.safetensors
  Total size                      33,799,602 bytes (32.23 MiB)
  Header length field             bytes 0..8 (8-byte unsigned little-endian) = 11,482
  Header JSON                     bytes 8..11,490 -- 11,482 bytes (11.21 KiB)
  Data buffer                     bytes 11,490..33,799,602 -- 33,788,112 bytes (32.22 MiB)

====================================================================================================
INTEGRITY
====================================================================================================
  Header entries                  111
  Tensors                         110
  __metadata__ present            yes (17 keys)
  Unparsable header entries       0
  Duplicate header keys           0
  Header JSON trailing padding    0 bytes
  Data buffer coverage            33,788,112 of 33,788,112 bytes (100.0000%)
  Gaps                            none
  Overlaps                        none
  Header sorted by data_offsets   no (specification recommends sorted)
  Size/shape/dtype agreement      110 agree
```

`__METADATA__`, with a JSON-encoded value expanded in place:

```
{
  "architecture": {  /* stored as a JSON-encoded string, shown decoded */
    "activation_function": "relu",
    "compute_data_type": "bfloat16",
    "input_encoding": "basic30",
    "value_head_style": "wdl_softmax"
  },
  "built_by_git": "085356f",
  "training_step": "5000"
}
```

Under `--verbose`, each segment is described individually:

```
  stem.conv.weight
  dtype                       F32 (32 bits per element)
  shape                       128x30x7x7  (188,160 elements)
  data_offsets                0..752,640
  absolute file offsets       11,490..764,130
  declared size               752,640 bytes (735.00 KiB)
  size from shape/dtype       752,640 bytes (735.00 KiB) -- matches
  first elements              [0.0397949, 0.0158691, -0.00340271, 0.00775146, ...]
    first 32 bytes:
                 0  00 00 23 3d 00 00 82 3c 00 00 5f bb 00 00 fe 3b  |..#=...<.._....;|
                16  00 00 f2 bb 00 00 01 3c 00 00 f0 3b 00 00 bb 3b  |.......<...;...;|
```

## What gets checked

Deviations from the [safetensors specification][spec] are reported in the `ISSUES`
section rather than aborting the dump, so a damaged file is still described as fully
as possible.

Only damage that makes the header unreadable stops the run (exit 3):

- The file is too short to hold the 8-byte length field
- The header runs past the end of the file
- The header is not valid UTF-8, not valid JSON, or not a JSON object
- The declared header size exceeds the 100 MB limit the reference implementation
  enforces — refused before the read, since the declared size is untrusted input and
  honouring it would allocate that much memory

Everything else is reported and the dump continues (exit 1):

- Header does not begin with `{` (0x7B)
- Header padding contains bytes other than spaces (0x20)
- Duplicate keys in the header
- `__metadata__` values that are not strings
- Entries missing or malforming `dtype`, `shape` or `data_offsets`
- Shapes with negative or non-integer dimensions
- dtypes the format does not define
- `data_offsets` whose end precedes its begin, or that run past the data buffer
- Sizes that disagree with what the shape and dtype imply
- Holes in the data buffer, and regions claimed by more than one tensor
- Sub-byte dtypes whose element count is not a whole number of bytes
- Tensors not listed in ascending `data_offsets` order (a warning; the specification
  recommends sorting but readers tolerate it)

## Supported dtypes

All 22 the format defines, matching the `Dtype` enum in the reference Rust
implementation:

| Bits | dtypes |
| --- | --- |
| 4 | `F4` |
| 6 | `F6_E2M3`, `F6_E3M2` |
| 8 | `BOOL`, `U8`, `I8`, `F8_E5M2`, `F8_E4M3`, `F8_E8M0`, `F8_E4M3FNUZ`, `F8_E5M2FNUZ` |
| 16 | `I16`, `U16`, `F16`, `BF16` |
| 32 | `I32`, `U32`, `F32` |
| 64 | `C64`, `F64`, `I64`, `U64` |

`--verbose` decodes element values for every dtype with an exact Python
representation. The sub-byte micro-scaling formats and the FP8 variants have no such
representation, so their bytes are shown as hexadecimal rather than decoded into
misleading numbers.

## File format reference

```
 8 bytes   N, an unsigned little-endian 64-bit integer: the header size
 N bytes   the header, a UTF-8 JSON object
 rest      the data buffer, with data_offsets measured from its start
```

## Development

```sh
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

## Releasing

Publishing to PyPI runs from `.github/workflows/publish.yml` using trusted publishing,
so no API token is stored here. It needs a one-time pending publisher configured on
PyPI (owner `drewster99`, repository `safetensors-print`, workflow `publish.yml`,
environment `pypi`).

After that, a release is: bump `__version__` in `src/safetensors_print/__init__.py`,
tag `vX.Y.Z`, and publish a GitHub release. The workflow runs the suite on 3.9 and
3.13, builds, verifies the tag matches the packaged version, and uploads.

## License

MIT — see [LICENSE](LICENSE).

[spec]: https://github.com/huggingface/safetensors#format
