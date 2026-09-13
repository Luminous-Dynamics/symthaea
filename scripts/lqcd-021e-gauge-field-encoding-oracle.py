#!/usr/bin/env python3
"""Independent canonical Wilson gauge-field byte-encoding oracle for LQCD-021E.

This subject freezes persistence bytes only.  It imports no Symthaea/Rust code
and uses exact binary64 fixture entries (0, +/-1), avoiding transcendental
cross-language assumptions.
"""
import hashlib
import json
import math
import struct

ORACLE_ID = "wilson_gauge_field_be_f64_v1"
TAG = b"symthaea.lqcd.wilson-gauge-field.v1\0"
SCALAR_BYTES = 8
MATRIX_COMPLEX_COMPONENTS = 18
DIRECTIONS = 4


def sha256(data):
    return hashlib.sha256(data).digest()


def bits(value):
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def f64_from_bits(value):
    return struct.unpack(">d", struct.pack(">Q", value))[0]


def identity_matrix():
    return (
        ((1.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        ((0.0, 0.0), (1.0, 0.0), (0.0, 0.0)),
        ((0.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
    )


def exact_diag(a, b, c):
    return (
        (a, (0.0, 0.0), (0.0, 0.0)),
        ((0.0, 0.0), b, (0.0, 0.0)),
        ((0.0, 0.0), (0.0, 0.0), c),
    )


def make_field(dims):
    x, y, z, t = dims
    if min(dims) <= 0:
        raise ValueError("zero extent")
    return [
        [
            [
                [
                    [identity_matrix() for _mu in range(DIRECTIONS)]
                    for _t in range(t)
                ]
                for _z in range(z)
            ]
            for _y in range(y)
        ]
        for _x in range(x)
    ]


def validate_dims(dims):
    if len(dims) != 4 or any(type(n) is not int or not 0 < n < (1 << 32) for n in dims):
        raise ValueError("invalid dimensions")


def encode(field, dims):
    validate_dims(dims)
    out = bytearray(TAG)
    for extent in dims:
        out += extent.to_bytes(4, "big")

    # Frozen order: x, y, z, t, mu, row, col, re, im.
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                for t in range(dims[3]):
                    for mu in range(DIRECTIONS):
                        matrix = field[x][y][z][t][mu]
                        for row in range(3):
                            for col in range(3):
                                re, im = matrix[row][col]
                                for value in (re, im):
                                    if not isinstance(value, float) or not math.isfinite(value):
                                        raise ValueError("non-finite or non-binary64 value")
                                    out += struct.pack(">Q", bits(value))
    return bytes(out)


def expected_length(dims):
    validate_dims(dims)
    sites = math.prod(dims)
    return len(TAG) + 16 + sites * DIRECTIONS * MATRIX_COMPLEX_COMPONENTS * SCALAR_BYTES


def decode(data):
    if not isinstance(data, bytes):
        raise TypeError("bytes required")
    if not data.startswith(TAG):
        raise ValueError("bad tag")
    cursor = len(TAG)
    if len(data) < cursor + 16:
        raise ValueError("truncated header")
    dims = tuple(int.from_bytes(data[cursor + 4*i:cursor + 4*i + 4], "big") for i in range(4))
    cursor += 16
    validate_dims(dims)
    if len(data) != expected_length(dims):
        raise ValueError("length mismatch")

    field = make_field(dims)
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                for t in range(dims[3]):
                    for mu in range(DIRECTIONS):
                        rows = []
                        for row in range(3):
                            cols = []
                            for col in range(3):
                                pair = []
                                for _component in range(2):
                                    raw = int.from_bytes(data[cursor:cursor+8], "big")
                                    cursor += 8
                                    value = f64_from_bits(raw)
                                    if not math.isfinite(value):
                                        raise ValueError("non-finite payload")
                                    pair.append(value)
                                cols.append(tuple(pair))
                            rows.append(tuple(cols))
                        field[x][y][z][t][mu] = tuple(rows)
    if cursor != len(data):
        raise AssertionError("decoder cursor mismatch")
    return field, dims


def field_bits(field, dims):
    result = []
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                for t in range(dims[3]):
                    for mu in range(DIRECTIONS):
                        matrix = field[x][y][z][t][mu]
                        for row in range(3):
                            for col in range(3):
                                re, im = matrix[row][col]
                                result.extend((bits(re), bits(im)))
    return tuple(result)


def main():
    dims = (2, 1, 1, 1)
    identity = make_field(dims)
    identity_bytes = encode(identity, dims)
    if len(identity_bytes) != expected_length(dims):
        raise AssertionError("identity length")
    decoded_identity, decoded_dims = decode(identity_bytes)
    if decoded_dims != dims or field_bits(decoded_identity, dims) != field_bits(identity, dims):
        raise AssertionError("identity roundtrip")

    fixture = make_field(dims)
    # Exact SU(3) matrices using only {0,+/-1,+/-i}; no libm dependency.
    fixture[0][0][0][0][1] = exact_diag(
        (0.0, 1.0), (0.0, -1.0), (1.0, 0.0)
    )
    fixture[1][0][0][0][2] = exact_diag(
        (-1.0, 0.0), (-1.0, 0.0), (1.0, 0.0)
    )
    fixture[1][0][0][0][3] = exact_diag(
        (0.0, -1.0), (0.0, 1.0), (1.0, 0.0)
    )
    fixture_bytes = encode(fixture, dims)
    decoded_fixture, decoded_dims = decode(fixture_bytes)
    if decoded_dims != dims or field_bits(decoded_fixture, dims) != field_bits(fixture, dims):
        raise AssertionError("fixture roundtrip")
    if encode(decoded_fixture, decoded_dims) != fixture_bytes:
        raise AssertionError("roundtrip bytes changed")

    # Link/site order is authoritative: moving an exact matrix to another link
    # must change the content digest even though the multiset of values is equal.
    reordered = make_field(dims)
    reordered[0][0][0][0][2] = fixture[0][0][0][0][1]
    reordered[1][0][0][0][1] = fixture[1][0][0][0][2]
    reordered[1][0][0][0][3] = fixture[1][0][0][0][3]
    reordered_bytes = encode(reordered, dims)
    if sha256(reordered_bytes) == sha256(fixture_bytes):
        raise AssertionError("ordering change did not alter digest")

    # Negative zero is preserved bit-for-bit by the scalar encoding.
    negzero = make_field((1, 1, 1, 1))
    m = [list(row) for row in negzero[0][0][0][0][0]]
    m[0] = list(m[0])
    m[0][1] = (-0.0, 0.0)
    negzero[0][0][0][0][0] = tuple(tuple(row) for row in m)
    negzero_bytes = encode(negzero, (1, 1, 1, 1))
    negzero_roundtrip, _ = decode(negzero_bytes)
    if bits(negzero_roundtrip[0][0][0][0][0][0][1][0]) != 0x8000000000000000:
        raise AssertionError("negative zero not preserved")

    for bad in (
        fixture_bytes[:-1],
        fixture_bytes + b"\0",
        b"wrong\0" + fixture_bytes[len(TAG):],
    ):
        try:
            decode(bad)
            raise AssertionError("malformed bytes accepted")
        except ValueError:
            pass

    # Zero dimensions fail at header interpretation.
    bad_dims = TAG + (0).to_bytes(4, "big") + (1).to_bytes(4, "big") * 3
    try:
        decode(bad_dims)
        raise AssertionError("zero dimension accepted")
    except ValueError:
        pass

    # Non-finite scalar payloads are not canonical field bytes.
    bad_nonfinite = bytearray(identity_bytes)
    payload_start = len(TAG) + 16
    bad_nonfinite[payload_start:payload_start+8] = struct.pack(">Q", 0x7ff8000000000000)
    try:
        decode(bytes(bad_nonfinite))
        raise AssertionError("NaN payload accepted")
    except ValueError:
        pass

    result = {
        "oracle_id": ORACLE_ID,
        "tag": TAG.decode("ascii").rstrip("\0"),
        "order": ["x", "y", "z", "t", "mu", "row", "col", "re", "im"],
        "dimension_encoding": "4xu32_big_endian",
        "scalar_encoding": "raw_ieee754_binary64_bits_big_endian",
        "fixture_dims": dims,
        "bytes_per_link": MATRIX_COMPLEX_COMPONENTS * SCALAR_BYTES,
        "identity_length": len(identity_bytes),
        "identity_sha256": sha256(identity_bytes).hex(),
        "fixture_length": len(fixture_bytes),
        "fixture_sha256": sha256(fixture_bytes).hex(),
        "reordered_sha256": sha256(reordered_bytes).hex(),
        "negative_zero_preserved": True,
        "roundtrip_exact_bits": True,
        "malformed_length_rejected": True,
        "zero_extent_rejected": True,
        "nonfinite_rejected": True,
    }
    text = json.dumps(result, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(text.encode()).hexdigest()
    print("ok")
    print("result_sha256=" + digest)
    print(text)


if __name__ == "__main__":
    main()
