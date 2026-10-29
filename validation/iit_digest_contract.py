# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-language v0.1 digest contract shared with crates/core/symthaea-iit.

This module has no PyPhi dependency so golden digest vectors can be tested in a
minimal Python environment. Any byte-format change requires a contract version bump.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence

MODEL_MAGIC = b"SYMTHAEA-IIT-BINARY-TPM\0v0.1\0"
EVALUATION_MAGIC = b"SYMTHAEA-IIT-EVALUATION\0v0.1\0"
ANALYSIS_REQUEST_MAGIC = b"SYMTHAEA-IIT-ANALYSIS-REQUEST\0v0.1\0"
MAX_DENSE_UNITS = 16


def canonical_model_bytes(
    tpm: Sequence[Sequence[float]], connectivity: Sequence[Sequence[int | bool]]
) -> bytes:
    """Encode the Rust BinaryCausalModel v0.1 binary wire contract."""
    n_units = len(tpm[0]) if tpm else 0
    if not 1 <= n_units <= MAX_DENSE_UNITS or len(tpm) != (1 << n_units):
        raise ValueError("TPM dimensions are outside the Rust v0.1 dense binary model contract")
    if any(len(row) != n_units for row in tpm):
        raise ValueError("TPM must be state-by-unit with exactly n_units columns")
    if len(connectivity) != n_units or any(len(row) != n_units for row in connectivity):
        raise ValueError("connectivity matrix shape does not match TPM unit count")

    encoded = bytearray(MODEL_MAGIC)
    encoded.extend(struct.pack("<I", n_units))
    for row in tpm:
        for cell in row:
            probability = float(cell)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("TPM contains a non-finite or out-of-range probability")
            if probability == 0.0:
                probability = 0.0
            encoded.extend(struct.pack("<d", probability))

    for row in connectivity:
        for edge in row:
            bit = int(edge)
            if bit not in (0, 1):
                raise ValueError("connectivity matrix must contain only binary entries")
            encoded.append(bit)
    return bytes(encoded)


def model_digest(model_bytes: bytes) -> str:
    return hashlib.sha256(model_bytes).hexdigest()


def evaluation_digest(model_bytes: bytes, state: Sequence[int | bool]) -> str:
    if any(int(bit) not in (0, 1) for bit in state):
        raise ValueError("evaluation state must contain only binary bits")
    return hashlib.sha256(EVALUATION_MAGIC + model_bytes + bytes(map(int, state))).hexdigest()


def analysis_request_digest(
    model_bytes: bytes,
    state: Sequence[int | bool],
    subset: Sequence[int],
    formalism: str,
    quantity_id: str,
) -> str:
    """Match AnalysisRequest::request_sha256 in the Rust crate byte-for-byte."""
    if any(int(bit) not in (0, 1) for bit in state):
        raise ValueError("evaluation state must contain only binary bits")
    if not subset or len(set(subset)) != len(subset):
        raise ValueError("candidate subsystem must be non-empty and contain unique unit indices")
    if any(int(unit) < 0 or int(unit) > 0xFFFF_FFFF for unit in subset):
        raise ValueError("candidate subsystem index is outside u32")
    encoded = bytearray(ANALYSIS_REQUEST_MAGIC)
    encoded.extend(model_bytes)
    encoded.extend(bytes(map(int, state)))
    encoded.extend(struct.pack("<I", len(subset)))
    for unit in subset:
        encoded.extend(struct.pack("<I", int(unit)))
    encoded.extend(b"\0")
    encoded.extend(formalism.encode("utf-8"))
    encoded.extend(b"\0")
    encoded.extend(quantity_id.encode("utf-8"))
    return hashlib.sha256(encoded).hexdigest()
