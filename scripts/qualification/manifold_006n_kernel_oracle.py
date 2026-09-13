#!/usr/bin/env python3
"""Independent exact-rational oracle for the MANIFOLD-006N N-A kernel.

The Rust harness emits only raw IEEE-754 binary64 bit patterns. This checker uses
Python's exact Fraction representation and does not import or reproduce the Rust
rounding implementation; it verifies that every emitted enclosure contains the
exact-real operation result and that every emitted scalar rejection is justified by
the declared finite-binary64 domain or division-by-zero boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from fractions import Fraction
from pathlib import Path

DOMAIN = b"symthaea.manifold-006n.kernel-vectors.v2\0"
MAX_FINITE_BITS = "7fefffffffffffff"


def binary64(hex_bits: str) -> float:
    raw = int(hex_bits, 16)
    if raw < 0 or raw > 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError(f"invalid binary64 bits: {hex_bits}")
    return struct.unpack(">d", raw.to_bytes(8, "big"))[0]


def exact(hex_bits: str) -> Fraction:
    value = binary64(hex_bits)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite binary64 input: {hex_bits}")
    return Fraction.from_float(value)


def operation_kind(op: str) -> str:
    for kind in ("add", "sub", "mul", "div"):
        if op.startswith(kind):
            return kind
    raise AssertionError(f"unsupported scalar operation: {op}")


def scalar_exact_result(op: str, left: Fraction, right: Fraction) -> Fraction:
    kind = operation_kind(op)
    if kind == "add":
        return left + right
    if kind == "sub":
        return left - right
    if kind == "mul":
        return left * right
    if right == 0:
        raise ZeroDivisionError(op)
    return left / right


def expected_interval(
    op: str,
    a_lower: Fraction,
    a_upper: Fraction,
    b_lower: Fraction,
    b_upper: Fraction,
) -> tuple[Fraction, Fraction]:
    if op.startswith("add") or op == "interval-add":
        return a_lower + b_lower, a_upper + b_upper
    if op.startswith("sub") or op == "interval-sub":
        return a_lower - b_upper, a_upper - b_lower
    if op.startswith("mul"):
        products = (
            a_lower * b_lower,
            a_lower * b_upper,
            a_upper * b_lower,
            a_upper * b_upper,
        )
        return min(products), max(products)
    if op == "interval-mul-positive":
        assert b_lower == b_upper and b_lower > 0
        return a_lower * b_lower, a_upper * b_lower
    if op.startswith("div"):
        assert b_lower == b_upper and b_lower != 0
        quotient = b_lower
        values = (a_lower / quotient, a_upper / quotient)
        return min(values), max(values)
    if op == "interval-div-positive":
        assert b_lower == b_upper and b_lower > 0
        return a_lower / b_lower, a_upper / b_lower
    raise AssertionError(f"unsupported operation: {op}")


def push_text(hasher: object, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("vectors", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()

    payload = args.vectors.read_bytes()
    lines = args.vectors.read_text().splitlines()
    if lines[:2] != [
        "schema\tsymthaea.manifold-006n.kernel-vectors.v3",
        "encoding\tieee754-binary64-u64-hex",
    ]:
        raise AssertionError("unexpected kernel vector header")

    vectors: list[dict[str, object]] = []
    rejections: list[dict[str, object]] = []
    gates: dict[str, str] = {}
    metadata: dict[str, int] = {}
    semantic = hashlib.sha3_256()
    semantic.update(DOMAIN)
    sweep_record_count = 0
    sweep_kinds: set[str] = set()
    max_finite = exact(MAX_FINITE_BITS)

    for line in lines[2:]:
        parts = line.split("\t")
        if parts[0] == "vector":
            if len(parts) != 8:
                raise AssertionError(f"malformed vector line: {line}")
            _, op, a_lo_b, a_hi_b, b_lo_b, b_hi_b, out_lo_b, out_hi_b = parts
            for raw in (a_lo_b, a_hi_b, b_lo_b, b_hi_b, out_lo_b, out_hi_b):
                if len(raw) != 16:
                    raise AssertionError(f"non-canonical bit width: {raw}")
            a_lo, a_hi = exact(a_lo_b), exact(a_hi_b)
            b_lo, b_hi = exact(b_lo_b), exact(b_hi_b)
            out_lo, out_hi = exact(out_lo_b), exact(out_hi_b)
            if a_lo > a_hi or b_lo > b_hi or out_lo > out_hi:
                raise AssertionError(f"{op}: reversed interval")
            exact_lo, exact_hi = expected_interval(op, a_lo, a_hi, b_lo, b_hi)
            if not (out_lo <= exact_lo <= exact_hi <= out_hi):
                raise AssertionError(f"{op}: Rust output does not contain exact result")

            semantic.update(b"V")
            push_text(semantic, op)
            for raw in (a_lo_b, a_hi_b, b_lo_b, b_hi_b, out_lo_b, out_hi_b):
                semantic.update(int(raw, 16).to_bytes(8, "big"))
            vectors.append(
                {
                    "operation": op,
                    "exact_result_contained": True,
                    "output_is_point": out_lo == out_hi,
                }
            )
            if "-sweep-" in op:
                sweep_record_count += 1
                sweep_kinds.add(operation_kind(op))
        elif parts[0] == "reject":
            if len(parts) != 5:
                raise AssertionError(f"malformed rejection line: {line}")
            _, op, left_b, right_b, reason = parts
            if len(left_b) != 16 or len(right_b) != 16:
                raise AssertionError(f"non-canonical rejection operand width: {line}")
            left, right = exact(left_b), exact(right_b)
            if reason == "division-by-zero":
                if operation_kind(op) != "div" or right != 0:
                    raise AssertionError(f"{op}: unjustified division-by-zero rejection")
            elif reason == "finite-domain":
                if operation_kind(op) == "div" and right == 0:
                    raise AssertionError(f"{op}: zero divisor mislabeled as finite-domain rejection")
                exact_result = scalar_exact_result(op, left, right)
                if abs(exact_result) <= max_finite:
                    raise AssertionError(
                        f"{op}: finite-domain rejection although exact result is representable in declared magnitude domain"
                    )
            else:
                raise AssertionError(f"{op}: unknown rejection reason {reason}")

            semantic.update(b"R")
            push_text(semantic, op)
            semantic.update(int(left_b, 16).to_bytes(8, "big"))
            semantic.update(int(right_b, 16).to_bytes(8, "big"))
            push_text(semantic, reason)
            rejections.append(
                {
                    "operation": op,
                    "reason": reason,
                    "exact_rejection_justified": True,
                }
            )
            if "-sweep-" in op:
                sweep_record_count += 1
                sweep_kinds.add(operation_kind(op))
        elif parts[0] == "meta":
            if len(parts) != 3:
                raise AssertionError(f"malformed metadata line: {line}")
            _, name, value = parts
            if name in metadata:
                raise AssertionError(f"duplicate metadata field: {name}")
            parsed = int(value)
            if parsed <= 0:
                raise AssertionError(f"non-positive metadata value: {line}")
            metadata[name] = parsed
            semantic.update(b"M")
            push_text(semantic, name)
            semantic.update(parsed.to_bytes(8, "big"))
        elif parts[0] == "gate":
            if len(parts) != 3:
                raise AssertionError(f"malformed gate line: {line}")
            _, name, verdict = parts
            if verdict != "PASS":
                raise AssertionError(f"kernel harness gate failed: {name}={verdict}")
            if name in gates:
                raise AssertionError(f"duplicate gate: {name}")
            gates[name] = verdict
            semantic.update(b"G")
            push_text(semantic, name)
            push_text(semantic, verdict)
        else:
            raise AssertionError(f"unknown vector record: {line}")

    required_gates = {
        "signed-zero-canonical",
        "ordered-signed-zero-preserved",
        "neighbor-zero-subnormal",
        "finite-edge-rejection",
        "nonfinite-result-rejection",
        "precondition-rejection",
        "qualified-error-text-preserved",
        "nonzero-underflow-enclosure",
    }
    if set(gates) != required_gates:
        raise AssertionError(f"kernel gate set mismatch: {set(gates)!r}")

    adversarial_value_count = metadata.get("adversarial-value-count")
    if adversarial_value_count is None or adversarial_value_count < 24:
        raise AssertionError("adversarial scalar corpus is unexpectedly small")
    expected_sweep_records = adversarial_value_count * adversarial_value_count * 4
    if sweep_record_count != expected_sweep_records:
        raise AssertionError(
            f"incomplete adversarial sweep: got {sweep_record_count}, expected {expected_sweep_records}"
        )
    if sweep_kinds != {"add", "sub", "mul", "div"}:
        raise AssertionError(f"incomplete sweep operation set: {sweep_kinds!r}")
    if not rejections:
        raise AssertionError("adversarial sweep produced no fail-closed rejection cases")

    by_op = {entry["operation"]: entry for entry in vectors}
    if not by_op["mul-underflow"]["exact_result_contained"]:
        raise AssertionError("underflow vector not contained")
    if not by_op["mul-power2"]["output_is_point"]:
        raise AssertionError("normal power-of-two multiply lost exact fast path")
    if not by_op["div-power2"]["output_is_point"]:
        raise AssertionError("normal power-of-two divide lost exact fast path")
    if not by_op["add-cancel"]["output_is_point"]:
        raise AssertionError("exact cancellation widened unexpectedly")

    receipt = {
        "schema": "symthaea.manifold-006n.kernel-exact-oracle.v3",
        "authority": "qualification-only-independent-python-exact-rational",
        "transport_sha256": hashlib.sha256(payload).hexdigest(),
        "semantic_vector_sha3_256": semantic.hexdigest(),
        "vector_count": len(vectors),
        "rejection_count": len(rejections),
        "gate_count": len(gates),
        "adversarial_value_count": adversarial_value_count,
        "sweep_record_count": sweep_record_count,
        "all_exact_results_contained": True,
        "all_scalar_rejections_exactly_justified": True,
        "deterministic_adversarial_sweep_complete": True,
        "normal_power_of_two_fast_paths_exact": True,
        "nonzero_underflow_outward_enclosed": True,
        "signed_zero_constructor_semantics_preserved": True,
        "nonfinite_results_fail_closed": True,
        "qualified_error_text_preserved": True,
        "production_runtime_authority": False,
        "verdict": "PASS",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
