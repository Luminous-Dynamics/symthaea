#!/usr/bin/env python3
"""Verify signed, non-authorizing Cyber Crucible verifier-decision receipts.

This module authenticates a decision under an externally supplied policy and binds
it to an independently supplied run context. It does not execute the evaluator,
prove the evidence true, or grant authority to perform an action.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

SCHEMA_VERSION = "cyber-crucible-verifier-decision.v1"
POLICY_VERSION = "cyber-crucible-decision-trust-policy.v1"
DOMAIN_SEPARATOR = b"symthaea-cyber-crucible-verifier-decision.v1\x00"
DECISION_ID = re.compile(r"^[0-9a-f]{32}$")
HEX_256 = re.compile(r"^[0-9a-f]{64}$")
B64URL = re.compile(r"^[A-Za-z0-9_-]+$")
MAX_DECISION_LIFETIME_SECONDS = 3600

PAYLOAD_FIELDS = {
    "schema_version", "algorithm", "decision_id", "decision_nonce",
    "signer_key_id", "verifier_id", "decision", "reason_code",
    "campaign_id", "subject_id", "attempt_id", "scenario_id",
    "scenario_revision", "scenario_digest", "input_receipt_digest", "report_digest",
    "evaluator_policy_digest", "context_digest", "verifier_build_digest",
    "verifier_provenance_policy_digest", "decision_trust_policy_digest", "issued_at", "expires_at",
    "non_authorizing",
}
CONTEXT_FIELDS = {
    "campaign_id", "subject_id", "attempt_id", "scenario_id",
    "scenario_revision", "scenario_digest", "input_receipt_digest", "report_digest",
    "evaluator_policy_digest", "verifier_build_digest",
    "verifier_provenance_policy_digest", "decision_nonce",
}
DECISIONS = {"accept", "reject"}
REASONS = {
    "authenticated_under_policy",
    "context_mismatch",
    "policy_rejected",
    "signature_invalid",
    "replay_rejected",
    "malformed_input",
    "expired_input",
    "verifier_error",
}


class DecisionVerificationError(ValueError):
    """The decision receipt cannot be authenticated under the supplied inputs."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DecisionVerificationError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def load_json(path: str | Path) -> Any:
    """Read strict UTF-8 JSON and reject duplicate members/non-JSON constants."""
    def reject_constant(value: str) -> None:
        raise DecisionVerificationError(f"non-JSON numeric constant: {value}")

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(
                handle, object_pairs_hook=_unique_object, parse_constant=reject_constant
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DecisionVerificationError(f"cannot read strict JSON input: {exc}") from exc


def canonical_json(value: Any) -> bytes:
    """RFC 8785 JSON Canonicalization Scheme bytes."""
    try:
        return rfc8785.dumps(value)
    except Exception as exc:
        raise DecisionVerificationError(f"value cannot be RFC 8785 canonicalized: {exc}") from exc


def canonical_json_digest(value: Any) -> str:
    """SHA-256 digest of the RFC 8785 representation of a JSON value."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def canonical_signed_bytes(payload: dict[str, Any]) -> bytes:
    """Domain-separated bytes signed by the decision signer."""
    return DOMAIN_SEPARATOR + canonical_json(payload)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise DecisionVerificationError(f"{label} must be a non-empty trimmed string")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX_256.fullmatch(value):
        raise DecisionVerificationError(f"{label} must be lowercase SHA-256 hex")
    return value


def _utc(value: Any, label: str) -> datetime:
    text = _text(value, label)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", text):
        raise DecisionVerificationError(f"{label} must use YYYY-MM-DDTHH:MM:SSZ")
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise DecisionVerificationError(f"{label} is not a valid UTC timestamp") from exc


def _b64url_decode(value: Any, label: str, expected_length: int) -> bytes:
    text = _text(value, label)
    if not B64URL.fullmatch(text) or "=" in text or len(text) % 4 == 1:
        raise DecisionVerificationError(f"{label} must be unpadded base64url")
    try:
        decoded = base64.urlsafe_b64decode(text + "=" * ((4 - len(text) % 4) % 4))
    except (binascii.Error, ValueError) as exc:
        raise DecisionVerificationError(f"{label} is malformed base64url") from exc
    if base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=") != text:
        raise DecisionVerificationError(f"{label} uses a non-canonical encoding")
    if len(decoded) != expected_length:
        raise DecisionVerificationError(f"{label} has the wrong decoded length")
    return decoded


def _validate_context(context: Any) -> dict[str, Any]:
    if not isinstance(context, dict) or set(context) != CONTEXT_FIELDS:
        raise DecisionVerificationError("expected context has missing or unknown fields")
    for field in ("campaign_id", "subject_id", "attempt_id", "scenario_id"):
        _text(context[field], f"expected context.{field}")
    revision = context["scenario_revision"]
    if not isinstance(revision, int) or isinstance(revision, bool) or not 1 <= revision <= 2**53 - 1:
        raise DecisionVerificationError("expected context.scenario_revision must be a safe positive integer")
    for field in (
        "scenario_digest", "input_receipt_digest", "report_digest", "evaluator_policy_digest",
        "verifier_build_digest", "verifier_provenance_policy_digest",
    ):
        _digest(context[field], f"expected context.{field}")
    _b64url_decode(context["decision_nonce"], "expected context.decision_nonce", 32)
    return context


def _validate_policy(policy: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "policy_id", "keys"}:
        raise DecisionVerificationError("decision trust policy has invalid top-level shape")
    if policy["schema_version"] != POLICY_VERSION:
        raise DecisionVerificationError("unsupported decision trust-policy schema_version")
    _text(policy["policy_id"], "decision trust policy policy_id")
    if not isinstance(policy["keys"], list) or not policy["keys"]:
        raise DecisionVerificationError("decision trust policy keys must be a non-empty list")

    by_id: dict[str, dict[str, Any]] = {}
    public_keys: set[bytes] = set()
    allowed = {
        "key_id", "verifier_id", "public_key", "not_before", "not_after", "revoked_at"
    }
    for index, entry in enumerate(policy["keys"]):
        label = f"decision trust policy keys[{index}]"
        if not isinstance(entry, dict) or set(entry) != allowed:
            raise DecisionVerificationError(f"{label} has invalid shape")
        key_id = _text(entry["key_id"], f"{label}.key_id")
        _text(entry["verifier_id"], f"{label}.verifier_id")
        if key_id in by_id:
            raise DecisionVerificationError(f"duplicate decision signer key_id: {key_id}")
        raw_key = _b64url_decode(entry["public_key"], f"{label}.public_key", 32)
        if raw_key in public_keys:
            raise DecisionVerificationError("duplicate public key material in decision trust policy")
        public_keys.add(raw_key)
        not_before = _utc(entry["not_before"], f"{label}.not_before")
        not_after = _utc(entry["not_after"], f"{label}.not_after")
        if not_before >= not_after:
            raise DecisionVerificationError(f"{label} has an empty or reversed validity window")
        if entry["revoked_at"] is not None:
            _utc(entry["revoked_at"], f"{label}.revoked_at")
        by_id[key_id] = entry
    return by_id


def _validate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_FIELDS:
        raise DecisionVerificationError("decision payload has missing or unknown fields")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise DecisionVerificationError("unsupported decision schema_version")
    if payload["algorithm"] != "Ed25519":
        raise DecisionVerificationError("only Ed25519 is accepted")
    if not isinstance(payload["decision_id"], str) or not DECISION_ID.fullmatch(payload["decision_id"]):
        raise DecisionVerificationError("decision_id must be 32 lowercase hexadecimal characters")
    _b64url_decode(payload["decision_nonce"], "payload.decision_nonce", 32)
    for field in (
        "signer_key_id", "verifier_id", "campaign_id", "subject_id", "attempt_id", "scenario_id"
    ):
        _text(payload[field], f"payload.{field}")
    if not isinstance(payload["decision"], str) or payload["decision"] not in DECISIONS:
        raise DecisionVerificationError("decision must be accept or reject")
    if not isinstance(payload["reason_code"], str) or payload["reason_code"] not in REASONS:
        raise DecisionVerificationError("decision reason_code is unsupported")
    if payload["decision"] == "accept" and payload["reason_code"] != "authenticated_under_policy":
        raise DecisionVerificationError("accept decisions require reason_code authenticated_under_policy")
    if payload["decision"] == "reject" and payload["reason_code"] == "authenticated_under_policy":
        raise DecisionVerificationError("reject decisions cannot use the acceptance reason code")
    revision = payload["scenario_revision"]
    if not isinstance(revision, int) or isinstance(revision, bool) or not 1 <= revision <= 2**53 - 1:
        raise DecisionVerificationError("scenario_revision must be a safe positive integer")
    for field in (
        "scenario_digest", "input_receipt_digest", "report_digest", "evaluator_policy_digest",
        "context_digest", "verifier_build_digest", "verifier_provenance_policy_digest",
        "decision_trust_policy_digest",
    ):
        _digest(payload[field], f"payload.{field}")
    issued_at = _utc(payload["issued_at"], "payload.issued_at")
    expires_at = _utc(payload["expires_at"], "payload.expires_at")
    lifetime = (expires_at - issued_at).total_seconds()
    if lifetime <= 0 or lifetime > MAX_DECISION_LIFETIME_SECONDS:
        raise DecisionVerificationError("decision lifetime must be positive and no more than one hour")
    if payload["non_authorizing"] is not True:
        raise DecisionVerificationError("decision receipt must be explicitly non-authorizing")
    return payload


class ReplayLedger:
    """Single-host replay fence; use a shared transactional store for distributed consumers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def consume(self, payload: dict[str, Any], now: datetime) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise DecisionVerificationError("decision replay ledger must not be a symlink")
        if not self.path.exists():
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(fd)
        if self.path.is_symlink() or not self.path.is_file():
            raise DecisionVerificationError("decision replay ledger must be a regular file")
        os.chmod(self.path, 0o600)
        try:
            db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            try:
                db.execute("PRAGMA busy_timeout=10000")
                db.execute(
                    """CREATE TABLE IF NOT EXISTS consumed_decisions (
                        decision_id TEXT PRIMARY KEY,
                        decision_nonce TEXT NOT NULL UNIQUE,
                        input_receipt_digest TEXT NOT NULL,
                        consumed_at TEXT NOT NULL
                    )"""
                )
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    """INSERT INTO consumed_decisions
                       (decision_id, decision_nonce, input_receipt_digest, consumed_at)
                       VALUES (?, ?, ?, ?)""",
                    (
                        payload["decision_id"], payload["decision_nonce"],
                        payload["input_receipt_digest"], now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    ),
                )
                db.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                try:
                    db.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise DecisionVerificationError("decision replay rejected: ID or nonce already consumed") from exc
            except Exception:
                try:
                    db.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                db.close()
        except sqlite3.Error as exc:
            raise DecisionVerificationError(f"decision replay ledger failure: {exc}") from exc


def verify_decision_receipt(
    receipt: Any,
    trust_policy: Any,
    expected_context: Any,
    *,
    now: datetime | None = None,
    replay_ledger: ReplayLedger | None = None,
) -> dict[str, Any]:
    """Authenticate one non-authorizing decision against an external policy/context.

    A returned decision of 'accept' means only that this signed decision is authentic
    under the supplied context and trust policy. It is not permission to execute actions.
    """
    if now is None:
        now = datetime.now(timezone.utc).replace(microsecond=0)
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise DecisionVerificationError("verification time must be timezone-aware")
    now = now.astimezone(timezone.utc)
    if replay_ledger is None:
        raise DecisionVerificationError("persistent replay ledger is mandatory")
    if not isinstance(receipt, dict) or set(receipt) != {"payload", "signature"}:
        raise DecisionVerificationError("decision envelope must contain exactly payload and signature")

    payload = _validate_payload(receipt["payload"])
    context = _validate_context(expected_context)
    keys = _validate_policy(trust_policy)
    policy_digest = canonical_json_digest(trust_policy)
    if payload["decision_trust_policy_digest"] != policy_digest:
        raise DecisionVerificationError("decision trust-policy digest mismatch")
    if payload["context_digest"] != canonical_json_digest(context):
        raise DecisionVerificationError("decision context digest mismatch")
    for field in CONTEXT_FIELDS:
        if payload.get(field) != context[field]:
            raise DecisionVerificationError(f"decision does not match expected context field: {field}")

    key_id = payload["signer_key_id"]
    if key_id not in keys:
        raise DecisionVerificationError("decision signer key is not trusted")
    key = keys[key_id]
    if payload["verifier_id"] != key["verifier_id"]:
        raise DecisionVerificationError("decision verifier identity does not match trusted key")
    issued_at = _utc(payload["issued_at"], "payload.issued_at")
    expires_at = _utc(payload["expires_at"], "payload.expires_at")
    key_start = _utc(key["not_before"], "trusted key not_before")
    key_end = _utc(key["not_after"], "trusted key not_after")
    if issued_at < key_start or issued_at >= key_end or now < key_start or now >= key_end:
        raise DecisionVerificationError("decision signer key is outside its validity window")
    if now < issued_at or now >= expires_at:
        raise DecisionVerificationError("decision receipt is not currently valid")
    if key["revoked_at"] is not None and now >= _utc(key["revoked_at"], "trusted key revoked_at"):
        raise DecisionVerificationError("decision signer key is revoked")

    public_key = _b64url_decode(key["public_key"], "trusted public_key", 32)
    signature = _b64url_decode(receipt["signature"], "decision signature", 64)
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature, canonical_signed_bytes(payload)
        )
    except InvalidSignature as exc:
        raise DecisionVerificationError("decision Ed25519 signature verification failed") from exc
    except (TypeError, ValueError) as exc:
        raise DecisionVerificationError(f"invalid Ed25519 verification input: {exc}") from exc

    # Consume replay state only after schema, context, policy, time, and signature checks.
    replay_ledger.consume(payload, now)
    return {
        "schema_version": "cyber-crucible-decision-verification-result.v1",
        "decision_receipt_verified": True,
        "decision": payload["decision"],
        "reason_code": payload["reason_code"],
        "decision_id": payload["decision_id"],
        "signer_key_id": payload["signer_key_id"],
        "verifier_id": payload["verifier_id"],
        "input_receipt_digest": payload["input_receipt_digest"],
        "report_digest": payload["report_digest"],
        "context_digest": payload["context_digest"],
        "evaluator_policy_digest": payload["evaluator_policy_digest"],
        "decision_trust_policy_digest": payload["decision_trust_policy_digest"],
        "verifier_build_digest": payload["verifier_build_digest"],
        "verifier_provenance_policy_digest": payload["verifier_provenance_policy_digest"],
        "campaign_id": payload["campaign_id"],
        "subject_id": payload["subject_id"],
        "attempt_id": payload["attempt_id"],
        "scenario_id": payload["scenario_id"],
        "scenario_revision": payload["scenario_revision"],
        "replay_checked": True,
        "non_authorizing": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, help="signed decision receipt JSON")
    parser.add_argument("--trust-policy", required=True, help="operator-controlled decision signer policy JSON")
    parser.add_argument("--expected-context", required=True, help="independently supplied expected context JSON")
    parser.add_argument("--replay-db", required=True, help="protected persistent SQLite replay ledger")
    args = parser.parse_args(argv)
    try:
        result = verify_decision_receipt(
            load_json(args.receipt),
            load_json(args.trust_policy),
            load_json(args.expected_context),
            replay_ledger=ReplayLedger(args.replay_db),
        )
        # Exit zero means the decision receipt was authenticated; consumers MUST still
        # inspect result["decision"]. A valid signed rejection is also a verified receipt.
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except (DecisionVerificationError, sqlite3.Error, OSError) as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
