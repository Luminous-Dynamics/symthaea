#!/usr/bin/env python3
"""Fail-closed canonical binding between policy IDs, hypothesis IDs and a risk matrix.

The trusted manifest digest must come from a separately authenticated receipt or
signed release. This module provides deterministic integrity binding, not identity
authentication by itself.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from typing import Sequence
import numpy as np

SCHEMA = "symthaea.policy-risk-binding.v1"


class PolicyRiskBindingError(ValueError):
    """The proposed planner matrix does not bind to the trusted policy manifest."""


@dataclass(frozen=True)
class PolicyRiskManifest:
    schema: str
    policy_ids: tuple[str, ...]
    hypothesis_ids: tuple[str, ...]
    shape: tuple[int, int]
    risk_matrix_sha256: str
    manifest_sha256: str

    def payload(self) -> dict:
        return {
            "schema": self.schema,
            "policy_ids": list(self.policy_ids),
            "hypothesis_ids": list(self.hypothesis_ids),
            "shape": list(self.shape),
            "risk_matrix_sha256": self.risk_matrix_sha256,
        }

    def to_dict(self) -> dict:
        return {**self.payload(), "manifest_sha256": self.manifest_sha256}


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _validate_ids(ids: Sequence[str], label: str) -> tuple[str, ...]:
    values = tuple(ids)
    if not values:
        raise PolicyRiskBindingError(f"{label}_empty")
    if any(not isinstance(v, str) or not v or v.strip() != v for v in values):
        raise PolicyRiskBindingError(f"{label}_invalid")
    if len(set(values)) != len(values):
        raise PolicyRiskBindingError(f"{label}_duplicate")
    return values


def _risk_array(risk_matrix: object, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    try:
        matrix = np.asarray(risk_matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PolicyRiskBindingError("risk_matrix_not_numeric") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise PolicyRiskBindingError("risk_matrix_invalid_shape")
    if expected_shape is not None and matrix.shape != expected_shape:
        raise PolicyRiskBindingError("risk_matrix_shape_mismatch")
    if not np.all(np.isfinite(matrix)):
        raise PolicyRiskBindingError("risk_matrix_nonfinite")
    if np.any(matrix < 0.0):
        raise PolicyRiskBindingError("risk_matrix_negative_loss")
    return matrix


def _matrix_digest(matrix: np.ndarray) -> str:
    # Explicit IEEE-754 binary64 big-endian canonical byte order across platforms.
    canonical = np.ascontiguousarray(matrix.astype(">f8", copy=False))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _manifest_digest(payload: dict) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def build_manifest(policy_ids: Sequence[str], hypothesis_ids: Sequence[str], risk_matrix: object) -> PolicyRiskManifest:
    policies = _validate_ids(policy_ids, "policy_ids")
    hypotheses = _validate_ids(hypothesis_ids, "hypothesis_ids")
    matrix = _risk_array(risk_matrix, (len(policies), len(hypotheses)))
    payload = {
        "schema": SCHEMA,
        "policy_ids": list(policies),
        "hypothesis_ids": list(hypotheses),
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "risk_matrix_sha256": _matrix_digest(matrix),
    }
    return PolicyRiskManifest(
        schema=SCHEMA,
        policy_ids=policies,
        hypothesis_ids=hypotheses,
        shape=(int(matrix.shape[0]), int(matrix.shape[1])),
        risk_matrix_sha256=payload["risk_matrix_sha256"],
        manifest_sha256=_manifest_digest(payload),
    )


def verify_binding(
    trusted_manifest: PolicyRiskManifest,
    trusted_manifest_sha256: str,
    runtime_policy_ids: Sequence[str],
    runtime_hypothesis_ids: Sequence[str],
    runtime_risk_matrix: object,
    execution_policy_ids: Sequence[str],
) -> str:
    """Return the trusted digest only if all ordered bindings match exactly.

    The caller must obtain trusted_manifest_sha256 from a separately
    authenticated release/receipt; accepting a digest supplied by the same
    untrusted source as the matrix defeats the trust boundary.
    """
    if trusted_manifest.schema != SCHEMA:
        raise PolicyRiskBindingError("manifest_schema_mismatch")
    if _manifest_digest(trusted_manifest.payload()) != trusted_manifest.manifest_sha256:
        raise PolicyRiskBindingError("manifest_self_digest_mismatch")
    if trusted_manifest.manifest_sha256 != trusted_manifest_sha256:
        raise PolicyRiskBindingError("manifest_trust_anchor_mismatch")

    policies = _validate_ids(runtime_policy_ids, "runtime_policy_ids")
    hypotheses = _validate_ids(runtime_hypothesis_ids, "runtime_hypothesis_ids")
    executed = _validate_ids(execution_policy_ids, "execution_policy_ids")
    if policies != trusted_manifest.policy_ids:
        raise PolicyRiskBindingError("runtime_policy_order_mismatch")
    if hypotheses != trusted_manifest.hypothesis_ids:
        raise PolicyRiskBindingError("runtime_hypothesis_order_mismatch")
    if executed != trusted_manifest.policy_ids:
        raise PolicyRiskBindingError("execution_policy_order_mismatch")
    matrix = _risk_array(runtime_risk_matrix, trusted_manifest.shape)
    if _matrix_digest(matrix) != trusted_manifest.risk_matrix_sha256:
        raise PolicyRiskBindingError("risk_matrix_digest_mismatch")
    return trusted_manifest.manifest_sha256
