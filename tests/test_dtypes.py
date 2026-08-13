"""The dtype registry must mirror the format's own definitions exactly."""

from __future__ import annotations

import math
import struct

import pytest

from safetensors_print.dtypes import DTYPES_BY_NAME, MAX_HEADER_SIZE, decoder_for, dtype_named

EXPECTED_BIT_SIZES = {
    "BOOL": 8,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "U8": 8,
    "I8": 8,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "C64": 64,
    "F64": 64,
    "I64": 64,
    "U64": 64,
}


def test_registry_matches_the_reference_implementation():
    assert set(DTYPES_BY_NAME) == set(EXPECTED_BIT_SIZES)
    for name, bits in EXPECTED_BIT_SIZES.items():
        assert DTYPES_BY_NAME[name].bits_per_element == bits


def test_maximum_header_size_matches_the_reference_implementation():
    assert MAX_HEADER_SIZE == 100_000_000


def test_sub_byte_flag_is_set_only_for_packed_dtypes():
    packed = {name for name, dtype in DTYPES_BY_NAME.items() if dtype.is_sub_byte}
    assert packed == {"F4", "F6_E2M3", "F6_E3M2"}


def test_unknown_dtype_name_returns_none():
    assert dtype_named("F128") is None
    assert dtype_named("f32") is None


def test_float32_decoder_round_trips():
    decode = decoder_for(dtype_named("F32"))
    assert decode(struct.pack("<3f", 1.5, -2.25, 0.0), 3) == [1.5, -2.25, 0.0]


def test_float64_decoder_round_trips():
    decode = decoder_for(dtype_named("F64"))
    assert decode(struct.pack("<2d", 1.0e300, -0.5), 2) == [1.0e300, -0.5]


def test_float16_decoder_round_trips():
    decode = decoder_for(dtype_named("F16"))
    assert decode(struct.pack("<2e", 1.5, -0.25), 2) == [1.5, -0.25]


def test_bfloat16_decoder_widens_to_single_precision():
    """bfloat16 holds the top 16 bits of an IEEE 754 single, so exactly-representable values survive."""
    decode = decoder_for(dtype_named("BF16"))
    halves = [struct.unpack("<I", struct.pack("<f", value))[0] >> 16 for value in (1.0, -2.5, 0.5)]
    raw = struct.pack("<3H", *halves)
    assert decode(raw, 3) == [1.0, -2.5, 0.5]


def test_bfloat16_decoder_handles_infinity():
    decode = decoder_for(dtype_named("BF16"))
    assert math.isinf(decode(struct.pack("<H", 0x7F80), 1)[0])


def test_complex64_decoder_pairs_two_floats():
    decode = decoder_for(dtype_named("C64"))
    assert decode(struct.pack("<4f", 1.0, 2.0, -3.0, 0.5), 2) == [complex(1.0, 2.0), complex(-3.0, 0.5)]


def test_bool_decoder_treats_any_nonzero_byte_as_true():
    decode = decoder_for(dtype_named("BOOL"))
    assert decode(b"\x00\x01\xff", 3) == [False, True, True]


def test_signed_and_unsigned_integers_decode_distinctly():
    assert decoder_for(dtype_named("I8"))(b"\xff", 1) == [-1]
    assert decoder_for(dtype_named("U8"))(b"\xff", 1) == [255]
    assert decoder_for(dtype_named("I64"))(struct.pack("<q", -5), 1) == [-5]
    assert decoder_for(dtype_named("U64"))(struct.pack("<Q", 2**63), 1) == [2**63]


def test_decoder_stops_at_the_requested_element_count():
    decode = decoder_for(dtype_named("F32"))
    assert decode(struct.pack("<4f", 1, 2, 3, 4), 2) == [1.0, 2.0]


@pytest.mark.parametrize("name", ["F4", "F6_E2M3", "F6_E3M2", "F8_E5M2", "F8_E4M3", "F8_E8M0", "F8_E4M3FNUZ", "F8_E5M2FNUZ"])
def test_dtypes_without_an_exact_python_form_have_no_decoder(name):
    """These are shown as hexadecimal rather than decoded into misleading numbers."""
    assert decoder_for(dtype_named(name)) is None
