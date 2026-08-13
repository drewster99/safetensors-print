"""The safetensors dtype registry.

Bit sizes and names mirror the `Dtype` enum in the reference Rust
implementation (huggingface/safetensors, `safetensors/src/tensor.rs`), which is
the normative definition of what a conforming file may contain.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence


@dataclass(frozen=True)
class DType:
    """A dtype the safetensors format defines.

    `bits_per_element` is the storage width, which is below 8 for the
    micro-scaling formats and therefore cannot be expressed in whole bytes.
    """

    name: str
    bits_per_element: int
    category: str
    description: str

    @property
    def is_sub_byte(self) -> bool:
        """Whether elements are packed more than one to a byte."""
        return self.bits_per_element < 8


_ALL_DTYPES: Sequence[DType] = (
    DType("BOOL", 8, "boolean", "Boolean, one byte per element"),
    DType("F4", 4, "float", "MXFP4 4-bit float (sub-byte, 2 elements per byte)"),
    DType("F6_E2M3", 6, "float", "MXFP6 6-bit float, 2-bit exponent, 3-bit mantissa (sub-byte)"),
    DType("F6_E3M2", 6, "float", "MXFP6 6-bit float, 3-bit exponent, 2-bit mantissa (sub-byte)"),
    DType("U8", 8, "uint", "Unsigned 8-bit integer"),
    DType("I8", 8, "int", "Signed 8-bit integer"),
    DType("F8_E5M2", 8, "float", "FP8 8-bit float, 5-bit exponent, 2-bit mantissa"),
    DType("F8_E4M3", 8, "float", "FP8 8-bit float, 4-bit exponent, 3-bit mantissa"),
    DType("F8_E8M0", 8, "float", "FP8 8-bit exponent-only scale factor"),
    DType("F8_E4M3FNUZ", 8, "float", "FP8 E4M3 finite-no-unsigned-zero variant"),
    DType("F8_E5M2FNUZ", 8, "float", "FP8 E5M2 finite-no-unsigned-zero variant"),
    DType("I16", 16, "int", "Signed 16-bit integer"),
    DType("U16", 16, "uint", "Unsigned 16-bit integer"),
    DType("F16", 16, "float", "IEEE 754 half precision"),
    DType("BF16", 16, "float", "bfloat16, truncated IEEE 754 single precision"),
    DType("I32", 32, "int", "Signed 32-bit integer"),
    DType("U32", 32, "uint", "Unsigned 32-bit integer"),
    DType("F32", 32, "float", "IEEE 754 single precision"),
    DType("C64", 64, "complex", "Complex number, two 32-bit floats"),
    DType("F64", 64, "float", "IEEE 754 double precision"),
    DType("I64", 64, "int", "Signed 64-bit integer"),
    DType("U64", 64, "uint", "Unsigned 64-bit integer"),
)

DTYPES_BY_NAME = {dtype.name: dtype for dtype in _ALL_DTYPES}

MAX_HEADER_SIZE = 100_000_000
"""Header byte limit enforced by the reference implementation."""


def dtype_named(name: str) -> Optional[DType]:
    """The dtype with this exact serialized name, or None if the format defines no such dtype."""
    return DTYPES_BY_NAME.get(name)


def _decode_bf16(raw: bytes, count: int) -> List[float]:
    """bfloat16 is the high half of an IEEE 754 single, so widening is a left shift."""
    return [
        struct.unpack("<f", struct.pack("<I", half << 16))[0]
        for (half,) in struct.iter_unpack("<H", raw[: count * 2])
    ]


def _struct_decoder(format_code: str) -> Callable[[bytes, int], List[object]]:
    element_size = struct.calcsize("<" + format_code)

    def decode(raw: bytes, count: int) -> List[object]:
        usable = raw[: count * element_size]
        return [value for (value,) in struct.iter_unpack("<" + format_code, usable)]

    return decode


def _decode_complex64(raw: bytes, count: int) -> List[complex]:
    return [
        complex(real, imaginary)
        for real, imaginary in struct.iter_unpack("<ff", raw[: count * 8])
    ]


def _decode_bool(raw: bytes, count: int) -> List[bool]:
    return [byte != 0 for byte in raw[:count]]


_ELEMENT_DECODERS = {
    "BOOL": _decode_bool,
    "U8": _struct_decoder("B"),
    "I8": _struct_decoder("b"),
    "I16": _struct_decoder("h"),
    "U16": _struct_decoder("H"),
    "F16": _struct_decoder("e"),
    "BF16": _decode_bf16,
    "I32": _struct_decoder("i"),
    "U32": _struct_decoder("I"),
    "F32": _struct_decoder("f"),
    "C64": _decode_complex64,
    "F64": _struct_decoder("d"),
    "I64": _struct_decoder("q"),
    "U64": _struct_decoder("Q"),
}


def decoder_for(dtype: DType) -> Optional[Callable[[bytes, int], List[object]]]:
    """A function turning raw little-endian bytes into element values.

    Returns None for dtypes with no unambiguous Python representation (the
    sub-byte micro-scaling formats and the FP8 variants), whose bytes are
    reported as hexadecimal instead of being decoded into misleading numbers.
    """
    return _ELEMENT_DECODERS.get(dtype.name)
