#!/usr/bin/env python3
"""Independent exact-rational oracle for MANIFOLD-006C1.

Consumes Rust-emitted IEEE-754 bit-pattern vectors. Production Rust arithmetic is
not imported or reimplemented. Every binary64 value is reconstructed from its raw
bits and promoted to Python Fraction exactly before theorem checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from fractions import Fraction
from pathlib import Path
from typing import Any

VECTOR_IDENTITY_DOMAIN = b"symthaea.manifold-006c1.vector-set.v1\0"
ENVELOPE_BIT_KEYS = (
    "initial_state_bits",
    "control_bits",
    "disturbance_min_bits",
    "disturbance_max_bits",
    "horizon_bits",
    "target_lower_bits",
    "target_upper_bits",
    "rust_envelope_lower_bits",
    "rust_envelope_upper_bits",
)
CLASSIFICATION_BIT_KEYS = (
    "control_min_bits",
    "control_max_bits",
    "initial_state_bits",
    "disturbance_min_bits",
    "disturbance_max_bits",
    "horizon_bits",
    "target_lower_bits",
    "target_upper_bits",
)


def binary64(hex_bits: str) -> float:
    bits = int(hex_bits, 16)
    if bits < 0 or bits > 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError(f"invalid binary64 bits: {hex_bits}")
    return struct.unpack(">d", bits.to_bytes(8, "big"))[0]


def exact(hex_bits: str) -> Fraction:
    value = binary64(hex_bits)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"oracle input must be finite: {hex_bits}")
    return Fraction.from_float(value)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def push_text(hasher: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)


def push_bits(hasher: Any, hex_bits: str) -> None:
    if len(hex_bits) != 16:
        raise AssertionError(f"non-canonical binary64 hex width: {hex_bits}")
    hasher.update(int(hex_bits, 16).to_bytes(8, "big"))


def semantic_vector_identity(vectors: dict[str, Any]) -> str:
    hasher = hashlib.sha3_256()
    hasher.update(VECTOR_IDENTITY_DOMAIN)
    envelopes = vectors["envelope_vectors"]
    classes = vectors["classification_vectors"]
    hasher.update(len(envelopes).to_bytes(8, "big"))
    for vector in envelopes:
        hasher.update(b"E")
        push_text(hasher, vector["name"])
        for key in ENVELOPE_BIT_KEYS:
            push_bits(hasher, vector[key])
    hasher.update(len(classes).to_bytes(8, "big"))
    for vector in classes:
        hasher.update(b"C")
        push_text(hasher, vector["name"])
        for key in CLASSIFICATION_BIT_KEYS:
            push_bits(hasher, vector[key])
        push_text(hasher, vector["rust_classification"])
        push_text(hasher, vector["rust_promoted_control_bits"])
    return hasher.hexdigest()


def verify_envelope(vector: dict[str, Any]) -> dict[str, Any]:
    x0 = exact(vector["initial_state_bits"])
    control = exact(vector["control_bits"])
    d_min = exact(vector["disturbance_min_bits"])
    d_max = exact(vector["disturbance_max_bits"])
    horizon = exact(vector["horizon_bits"])
    rust_lower = exact(vector["rust_envelope_lower_bits"])
    rust_upper = exact(vector["rust_envelope_upper_bits"])

    if horizon <= 0:
        raise AssertionError(f"{vector['name']}: non-positive horizon")
    if d_min > d_max:
        raise AssertionError(f"{vector['name']}: reversed disturbance interval")

    terminal_lower = x0 + horizon * (control + d_min)
    terminal_upper = x0 + horizon * (control + d_max)
    contained = rust_lower <= terminal_lower <= terminal_upper <= rust_upper
    if not contained:
        raise AssertionError(
            f"{vector['name']}: Rust enclosure does not contain exact terminal extrema"
        )

    return {
        "name": vector["name"],
        "exact_terminal_ordered": True,
        "rust_encloses_exact_terminal_extrema": True,
    }


def exact_robust_exists(vector: dict[str, Any]) -> bool:
    u_min = exact(vector["control_min_bits"])
    u_max = exact(vector["control_max_bits"])
    x0 = exact(vector["initial_state_bits"])
    d_min = exact(vector["disturbance_min_bits"])
    d_max = exact(vector["disturbance_max_bits"])
    horizon = exact(vector["horizon_bits"])
    target_lower = exact(vector["target_lower_bits"])
    target_upper = exact(vector["target_upper_bits"])

    if horizon <= 0:
        raise AssertionError(f"{vector['name']}: non-positive horizon")
    required_lower = (target_lower - x0) / horizon - d_min
    required_upper = (target_upper - x0) / horizon - d_max
    feasible_lower = max(required_lower, u_min)
    feasible_upper = min(required_upper, u_max)
    return feasible_lower <= feasible_upper


def verify_promoted_control(vector: dict[str, Any]) -> bool:
    promoted = vector["rust_promoted_control_bits"]
    if promoted == "none":
        return False

    control = exact(promoted)
    u_min = exact(vector["control_min_bits"])
    u_max = exact(vector["control_max_bits"])
    x0 = exact(vector["initial_state_bits"])
    d_min = exact(vector["disturbance_min_bits"])
    d_max = exact(vector["disturbance_max_bits"])
    horizon = exact(vector["horizon_bits"])
    target_lower = exact(vector["target_lower_bits"])
    target_upper = exact(vector["target_upper_bits"])

    if not (u_min <= control <= u_max):
        raise AssertionError(f"{vector['name']}: promoted Rust control is not admissible")
    exact_lower = x0 + horizon * (control + d_min)
    exact_upper = x0 + horizon * (control + d_max)
    if not (target_lower <= exact_lower <= exact_upper <= target_upper):
        raise AssertionError(
            f"{vector['name']}: promoted Rust control is not exactly robust for the target"
        )
    return True


def verify_classification(vector: dict[str, Any]) -> dict[str, Any]:
    exact_exists = exact_robust_exists(vector)
    rust = vector["rust_classification"]
    promoted_control_verified = False

    if rust == "RobustFeasible":
        if not exact_exists:
            raise AssertionError(f"{vector['name']}: false-positive RobustFeasible")
        promoted_control_verified = verify_promoted_control(vector)
        if not promoted_control_verified:
            raise AssertionError(f"{vector['name']}: RobustFeasible omitted promoted control")
    elif vector["rust_promoted_control_bits"] != "none":
        raise AssertionError(f"{vector['name']}: non-feasible result carried a promoted control")

    if rust == "CertifiedNotRobust" and exact_exists:
        raise AssertionError(f"{vector['name']}: false-positive CertifiedNotRobust")
    if rust not in {"RobustFeasible", "CertifiedNotRobust", "Unknown"}:
        raise AssertionError(f"{vector['name']}: unknown Rust classification {rust}")

    if vector["name"] == "robust-feasible":
        assert rust == "RobustFeasible" and exact_exists and promoted_control_verified
    elif vector["name"] == "rounding-boundary-unknown":
        assert rust == "Unknown" and not exact_exists
    elif vector["name"] == "certified-not-robust":
        assert rust == "CertifiedNotRobust" and not exact_exists

    return {
        "name": vector["name"],
        "exact_robust_exists": exact_exists,
        "rust_classification": rust,
        "promoted_control_exactly_verified": promoted_control_verified,
        "classification_noncontradiction": True,
    }


def named_arithmetic_checks(vectors: dict[str, Any]) -> dict[str, bool]:
    by_name = {v["name"]: v for v in vectors["envelope_vectors"]}

    decimal = by_name["decimal-rounding-trap"]
    d_max = exact(decimal["disturbance_max_bits"])
    horizon = exact(decimal["horizon_bits"])
    target_upper = exact(decimal["target_upper_bits"])
    decimal_trap = d_max * horizon > target_upper
    if not decimal_trap:
        raise AssertionError("decimal-rounding trap exact inequality was not reproduced")

    underflow = by_name["nonzero-underflow"]
    control = exact(underflow["control_bits"])
    tiny_horizon = exact(underflow["horizon_bits"])
    nonzero_underflow = control * tiny_horizon > 0
    if not nonzero_underflow:
        raise AssertionError("nonzero-underflow exact product was not positive")
    if binary64(underflow["control_bits"]) * binary64(underflow["horizon_bits"]) != 0.0:
        raise AssertionError("underflow fixture no longer rounds to stored binary64 zero")

    power = by_name["power-of-two-exact"]
    exact_power_terminal = (
        exact(power["initial_state_bits"])
        + exact(power["horizon_bits"])
        * (exact(power["control_bits"]) + exact(power["disturbance_min_bits"]))
    )
    power_of_two_exact = (
        exact_power_terminal == Fraction(1, 1)
        and exact(power["rust_envelope_lower_bits"]) == Fraction(1, 1)
        and exact(power["rust_envelope_upper_bits"]) == Fraction(1, 1)
    )
    if not power_of_two_exact:
        raise AssertionError("power-of-two exact fixture widened or changed unexpectedly")

    return {
        "decimal_rounding_trap_exact_inequality": True,
        "nonzero_underflow_exact_product_positive": True,
        "power_of_two_normal_case_exact": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("vectors", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()

    payload = args.vectors.read_bytes()
    vectors = json.loads(payload)
    if vectors.get("schema") != "symthaea.manifold-006c1.oracle-vectors.v2":
        raise AssertionError("unexpected vector schema")
    if vectors.get("bit_encoding") != "ieee754-binary64-u64-hex":
        raise AssertionError("unexpected bit encoding")
    if vectors.get("vector_identity_algorithm") != "sha3-256-domain-separated-semantic-v1":
        raise AssertionError("unexpected vector identity algorithm")

    semantic_identity = semantic_vector_identity(vectors)
    if vectors.get("vector_set_identity") != semantic_identity:
        raise AssertionError("Rust/Python semantic vector-set identity mismatch")

    envelope_results = [verify_envelope(v) for v in vectors["envelope_vectors"]]
    classification_results = [
        verify_classification(v) for v in vectors["classification_vectors"]
    ]
    named_checks = named_arithmetic_checks(vectors)

    receipt = {
        "schema": "symthaea.manifold-006c1.exact-rational-oracle.v3",
        "authority": "qualification-only-independent-python-exact-rational",
        "vector_sha256": sha256_bytes(payload),
        "vector_set_identity": semantic_identity,
        "vector_identity_algorithm": "sha3-256-domain-separated-semantic-v1",
        "semantic_identity_independently_recomputed": True,
        "promoted_positive_control_exactly_verified": True,
        "envelope_vector_count": len(envelope_results),
        "classification_vector_count": len(classification_results),
        "envelope_results": envelope_results,
        "classification_results": classification_results,
        **named_checks,
        "production_runtime_authority": False,
        "verdict": "PASS",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
