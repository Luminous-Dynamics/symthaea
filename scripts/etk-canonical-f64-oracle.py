#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent canonical binary64 identity oracle for ETK semantic hashes.

Canonical identity form:

    f64:<16 lowercase hexadecimal digits>

The hexadecimal payload is the big-endian IEEE-754 binary64 bit pattern after
normalizing both +0.0 and -0.0 to +0.0. NaN and infinities are rejected.

This is an identity/canonicalization primitive only. It does not make a numeric
value correct, unit-consistent, physically meaningful, or authoritative.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from typing import Any


class CanonicalFloatError(ValueError):
    pass


def canonical_f64(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CanonicalFloatError("value must be a JSON/RFC numeric scalar")
    try:
        binary64 = float(value)
    except (OverflowError, ValueError) as error:
        raise CanonicalFloatError("value is not representable as binary64") from error
    if not math.isfinite(binary64):
        raise CanonicalFloatError("NaN and infinity are not admissible")
    # Signed zero is not an engineering semantic distinction for ETK v1.
    # Normalize both bit patterns before identity construction.
    if binary64 == 0.0:
        binary64 = 0.0
    return "f64:" + struct.pack(">d", binary64).hex()


def self_test() -> None:
    vectors = {
        0.0: "f64:0000000000000000",
        1.0: "f64:3ff0000000000000",
        -2.5: "f64:c004000000000000",
        0.1: "f64:3fb999999999999a",
        5e-324: "f64:0000000000000001",
        1.7976931348623157e308: "f64:7fefffffffffffff",
        -1.7976931348623157e308: "f64:ffefffffffffffff",
    }
    for value, expected in vectors.items():
        actual = canonical_f64(value)
        assert actual == expected, (value, actual, expected)

    assert canonical_f64(-0.0) == canonical_f64(0.0)
    assert canonical_f64(1) == canonical_f64(1.0)

    for invalid in (float("nan"), float("inf"), float("-inf"), True, "1.0", None):
        try:
            canonical_f64(invalid)
        except CanonicalFloatError:
            pass
        else:
            raise AssertionError(f"expected rejection for {invalid!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("values", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("ok canonical-f64-v1")
        for value in [0.0, -0.0, 1.0, -2.5, 0.1, 5e-324, 1.7976931348623157e308]:
            print(f"{value!r} -> {canonical_f64(value)}")
        return 0

    if args.values:
        for raw in args.values:
            try:
                value = json.loads(raw)
                print(canonical_f64(value))
            except (json.JSONDecodeError, CanonicalFloatError) as error:
                print(f"deny: {error}", file=sys.stderr)
                return 2
        return 0

    try:
        value = json.load(sys.stdin)
        print(canonical_f64(value))
        return 0
    except (json.JSONDecodeError, CanonicalFloatError) as error:
        print(json.dumps({"decision": "Deny", "reason": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
