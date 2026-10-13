"""Fixed certificate vectors shared by wire-format and formal-check tests.

The byte sequences are protocol fixtures, not generated expectations. Keep them
literal so the Python verifier, Rust implementation, and Lean proof bridge can
all be tested against stable canonical bytes.
"""
from __future__ import annotations

SYMPPRF1_PRIME_3 = bytes.fromhex(
    "53594d5050524631"
    "0000000000000003"
    "00000002"
    "0000000000000002" "00000000" "00000000"
    "0000000000000003" "00000001"
    "0000000000000002" "00000001"
    "00000001" "0000000000000002" "0000000000000002"
)

SYMFCER1_FACTOR_2 = bytes.fromhex(
    "53594d4643455231"
    "0000000000000002"
    "00000001"
    "0000000000000002"
    "00000001"
    "00000001"
    "00000024"
    "53594d5050524631"
    "0000000000000002"
    "00000001"
    "0000000000000002"
    "00000000"
    "00000000"
)

assert len(SYMPPRF1_PRIME_3) == 80
assert len(SYMFCER1_FACTOR_2) == 76

# A bridge-only fixture that exercises two distinct prime factors of 13 - 1.
# It supplements, but does not replace, the two Rust/Python protocol golden vectors.
SYMPPRF1_PRIME_13_MULTI_FACTOR = bytes.fromhex(
    "53594d5050524631"
    "000000000000000d"
    "00000003"
    "0000000000000002" "00000000" "00000000"
    "0000000000000003" "00000001"
    "0000000000000002" "00000001"
    "00000001" "0000000000000002" "0000000000000002"
    "000000000000000d" "00000002"
    "0000000000000002" "00000002"
    "0000000000000003" "00000001"
    "00000002"
    "0000000000000002" "0000000000000002"
    "0000000000000003" "0000000000000002"
)

assert len(SYMPPRF1_PRIME_13_MULTI_FACTOR) == 152
