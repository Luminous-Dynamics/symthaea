#!/usr/bin/env python3
"""Independent REGEN-004B subject-identity framing oracle.

This standard-library oracle intentionally imports no Symthaea or Mycelix code.
It qualifies only the narrow textual framing contract frozen by Mycelix
REGEN-002/004A and must not be interpreted as authentication or authority.
"""

from __future__ import annotations

import hashlib
import json
import string
from pathlib import Path
from typing import Any

SCHEMA = "mycelix.regen.subject-id-golden-v1"
EXPECTED_FIXTURE_SHA256 = "23fd4603e283b271a33d566c6f1678cd05e22fdd522d1cb55b3fd0264eadf9ef"
MAX_LOCAL_TOKEN_BYTES = 128
KINDS = (
    "site",
    "soil-plot",
    "biomass-lot",
    "biochar-batch",
    "compost-batch",
    "co-composted-amendment-batch",
    "recipe",
    "field-trial",
    "treatment-arm",
    "facility",
    "project",
    "quality-profile",
    "evidence-bundle",
)
_ALLOWED = frozenset(string.ascii_lowercase + string.digits + "-_.")
_EDGE_ALLOWED = frozenset(string.ascii_lowercase + string.digits)


class IdentityError(ValueError):
    """The supplied subject identity violates the frozen REGEN v1 grammar."""


def validate_local_token(token: str) -> None:
    """Validate the kind-local token with byte-equivalent Mycelix semantics."""
    try:
        encoded = token.encode("ascii")
    except UnicodeEncodeError as exc:
        raise IdentityError("local token must be ASCII") from exc

    if not encoded:
        raise IdentityError("local token must not be empty")
    if len(encoded) > MAX_LOCAL_TOKEN_BYTES:
        raise IdentityError("local token exceeds 128 bytes")
    if token[0] not in _EDGE_ALLOWED or token[-1] not in _EDGE_ALLOWED:
        raise IdentityError("local token must begin/end with lowercase ASCII alphanumeric")
    if any(char not in _ALLOWED for char in token):
        raise IdentityError("local token contains a noncanonical character")


def canonical_id(kind: str, token: str) -> str:
    """Construct exact `regen:v1:<kind>:<token>` identity text."""
    if kind not in KINDS:
        raise IdentityError(f"unknown subject kind: {kind}")
    validate_local_token(token)
    return f"regen:v1:{kind}:{token}"


def parse_canonical(kind: str, value: str) -> str:
    """Parse exact canonical text and return the validated local token."""
    if kind not in KINDS:
        raise IdentityError(f"unknown subject kind: {kind}")
    prefix = f"regen:v1:{kind}:"
    if not value.startswith(prefix):
        raise IdentityError("wrong canonical prefix")
    token = value[len(prefix) :]
    validate_local_token(token)
    if canonical_id(kind, token) != value:
        raise IdentityError("noncanonical round trip")
    return token


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IdentityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_fixture() -> tuple[bytes, dict[str, Any]]:
    """Load exact checked-in fixture bytes and reject identity drift."""
    path = Path(__file__).resolve().parent / "fixtures" / "regen_subject_ids_v1.json"
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != EXPECTED_FIXTURE_SHA256:
        raise IdentityError(
            f"fixture SHA-256 drift: expected {EXPECTED_FIXTURE_SHA256}, got {actual}"
        )
    decoded = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(decoded, dict):
        raise IdentityError("fixture root must be an object")
    return raw, decoded


def self_test() -> dict[str, Any]:
    """Execute the independent valid/invalid golden-vector campaign."""
    raw, fixture = load_fixture()
    if fixture.get("schema") != SCHEMA:
        raise IdentityError("fixture schema mismatch")

    valid = fixture.get("valid")
    invalid = fixture.get("invalid")
    if not isinstance(valid, list) or not isinstance(invalid, list):
        raise IdentityError("fixture valid/invalid members must be arrays")

    observed_kinds = [row.get("kind") for row in valid if isinstance(row, dict)]
    if len(observed_kinds) != len(KINDS) or set(observed_kinds) != set(KINDS):
        raise IdentityError("valid-vector kind roster is not exact")

    seen_canonical: set[str] = set()
    for row in valid:
        if not isinstance(row, dict):
            raise IdentityError("valid vector must be an object")
        kind = row.get("kind")
        token = row.get("token")
        expected = row.get("canonical")
        if not all(isinstance(value, str) for value in (kind, token, expected)):
            raise IdentityError("valid vector fields must be strings")
        produced = canonical_id(kind, token)
        if produced != expected:
            raise IdentityError(f"valid vector mismatch: {kind} {token}")
        if parse_canonical(kind, expected) != token:
            raise IdentityError(f"valid parse mismatch: {expected}")
        if expected in seen_canonical:
            raise IdentityError(f"duplicate canonical identity: {expected}")
        seen_canonical.add(expected)

    for row in invalid:
        if not isinstance(row, dict):
            raise IdentityError("invalid vector must be an object")
        kind = row.get("kind")
        value = row.get("canonical")
        reason = row.get("reason")
        if not all(isinstance(item, str) and item for item in (kind, value, reason)):
            raise IdentityError("invalid vector fields must be nonempty strings")
        try:
            parse_canonical(kind, value)
        except IdentityError:
            pass
        else:
            raise IdentityError(
                f"invalid vector unexpectedly accepted: kind={kind} value={value} reason={reason}"
            )

    return {
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "invalid_vectors": len(invalid),
        "result": "ok",
        "schema": SCHEMA,
        "valid_vectors": len(valid),
    }


def main() -> None:
    """Run the frozen independent campaign and emit a deterministic summary."""
    print(json.dumps(self_test(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
