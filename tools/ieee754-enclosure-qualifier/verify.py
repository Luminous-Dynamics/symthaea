#!/usr/bin/env python3
"""Qualification-only exact-rational checker for Rust binary64 enclosures.

This script is never imported by or linked into production Rust code. It reads
raw u64 input/output bit patterns emitted by the Rust qualification binary and
checks every supported arithmetic result against Python's exact Fraction model.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import struct
import sys
from fractions import Fraction
from pathlib import Path


def f64_from_bits(hex_bits: str) -> float:
    if len(hex_bits) != 16:
        raise ValueError(f"expected 16 hex digits for binary64 bits, got {hex_bits!r}")
    raw = int(hex_bits, 16)
    return struct.unpack(">d", raw.to_bytes(8, "big", signed=False))[0]


def exact_operation(op: str, a: Fraction, b: Fraction) -> Fraction:
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        if b == 0:
            raise ZeroDivisionError("division by exact zero")
        return a / b
    raise ValueError(f"unsupported operation {op!r}")


def endpoint_contains_lower(endpoint: float, exact: Fraction) -> bool:
    if math.isnan(endpoint) or endpoint == math.inf:
        return False
    if endpoint == -math.inf:
        return True
    return Fraction.from_float(endpoint) <= exact


def endpoint_contains_upper(endpoint: float, exact: Fraction) -> bool:
    if math.isnan(endpoint) or endpoint == -math.inf:
        return False
    if endpoint == math.inf:
        return True
    return exact <= Fraction.from_float(endpoint)


def verify(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    previous_id = -1
    rejected_zero_divisors = 0
    overflow_enclosures = 0

    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                fields = raw_line.decode("ascii").rstrip("\n").split("|")
                if len(fields) != 7:
                    raise ValueError(f"expected 7 fields, got {len(fields)}")
                case_id_text, op, a_hex, b_hex, lower_hex, upper_hex, status = fields
                case_id = int(case_id_text)
                if case_id <= previous_id:
                    raise ValueError("case IDs must be strictly increasing")
                previous_id = case_id

                a_float = f64_from_bits(a_hex)
                b_float = f64_from_bits(b_hex)
                if not math.isfinite(a_float) or not math.isfinite(b_float):
                    raise ValueError("input vector contains a non-finite operand")
                a = Fraction.from_float(a_float)
                b = Fraction.from_float(b_float)

                if op == "div" and b == 0:
                    if status != "REJECTED_ZERO_DIVISOR":
                        raise ValueError("division by zero was not rejected")
                    if lower_hex or upper_hex:
                        raise ValueError("rejected division must not emit endpoints")
                    rejected_zero_divisors += 1
                    count += 1
                    continue

                if status != "OK":
                    raise ValueError(f"unexpected arithmetic rejection: {status}")
                if not lower_hex or not upper_hex:
                    raise ValueError("successful operation is missing endpoints")

                exact = exact_operation(op, a, b)
                lower = f64_from_bits(lower_hex)
                upper = f64_from_bits(upper_hex)
                if math.isnan(lower) or math.isnan(upper) or lower > upper:
                    raise ValueError("invalid endpoint order or NaN endpoint")
                if not endpoint_contains_lower(lower, exact):
                    raise AssertionError(
                        f"case {case_id}: lower endpoint {lower!r} is above exact {exact}"
                    )
                if not endpoint_contains_upper(upper, exact):
                    raise AssertionError(
                        f"case {case_id}: upper endpoint {upper!r} is below exact {exact}"
                    )
                if lower in (-math.inf, math.inf) or upper in (-math.inf, math.inf):
                    overflow_enclosures += 1
                count += 1
            except Exception as exc:  # noqa: BLE001 - qualification must fail visibly
                raise RuntimeError(f"{path}:{line_number}: {exc}") from exc

    if count < 1_000:
        raise RuntimeError(f"corpus unexpectedly small: only {count} cases")
    if rejected_zero_divisors < 1:
        raise RuntimeError("corpus did not exercise zero-divisor rejection")

    return count, digest.hexdigest()


def overflow_count(path: Path) -> int:
    count = 0
    with path.open("rb") as stream:
        for raw_line in stream:
            fields = raw_line.decode("ascii").rstrip("\n").split("|")
            if len(fields) == 7 and fields[6] == "OK":
                lower = f64_from_bits(fields[4])
                upper = f64_from_bits(fields[5])
                if math.isinf(lower) or math.isinf(upper):
                    count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("vectors", type=Path, help="Rust-emitted pipe-delimited vector file")
    args = parser.parse_args()

    try:
        count, corpus_digest = verify(args.vectors)
    except Exception as exc:  # noqa: BLE001 - CLI must return failure for any oracle defect
        print(f"EXACT_RATIONAL_QUALIFICATION=FAIL: {exc}", file=sys.stderr)
        return 1

    print("EXACT_RATIONAL_QUALIFICATION=PASS")
    print(f"CASE_COUNT={count}")
    print(f"VECTOR_CORPUS_SHA256={corpus_digest}")
    print(f"UNBOUNDED_OVERFLOW_CASES={overflow_count(args.vectors)}")
    print(f"PYTHON_VERSION={sys.version.split()[0]}")
    print("REFERENCE=Python Fraction.from_float over exact raw binary64 inputs")
    print("SCOPE=single-operation scalar add/sub/mul/div enclosures only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
