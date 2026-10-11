#!/usr/bin/env python3
"""Fail-closed Ed25519 verification for non-authorizing Cyber Crucible receipts.

Signatures cover RFC 8785 canonical JSON preceded by a versioned domain separator.
The verifier authenticates the receipt payload against an external trust policy and
uses a persistent SQLite uniqueness ledger for one-time receipt/nonce consumption.
It does not grant operational authority or prove that a reported action was safe.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import hashlib
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

SCHEMA_VERSION = "cyber-crucible-signed-receipt.v1"
POLICY_VERSION = "cyber-crucible-trust-policy.v1"
DOMAIN_SEPARATOR = b"symthaea-cyber-crucible-signed-receipt.v1\x00"
HEX_256 = re.compile(r"^[0-9a-f]{64}$")
RECEIPT_ID = re.compile(r"^[0-9a-f]{32}$")
B64URL = re.compile(r"^[A-Za-z0-9_-]+$")

PAYLOAD_FIELDS = {
    "schema_version", "algorithm", "receipt_id", "key_id", "evaluator_id",
    "role", "campaign_id", "subject_id", "attempt_id", "receipt_nonce", "issued_at", "expires_at",
    "scenario_id", "scenario_revision", "scenario_digest", "evidence_digest",
    "status", "proposal_id", "proposal_digest",
}
ROLES = {
    "functional_verdict", "security_verdict", "authorization_decision",
    "outcome_verification", "recovery_verification",
}
STATUSES = {"pass", "fail", "inconclusive", "not_run"}


class VerificationError(ValueError):
    """Input cannot be accepted as an authenticated, context-bound receipt."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """JSON object-pairs hook that rejects duplicate names before dict collapse."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def load_json(path: str | Path) -> Any:
    """Load UTF-8 JSON while rejecting duplicate object members and non-JSON constants."""
    def bad_constant(value: str) -> None:
        raise VerificationError(f"non-JSON numeric constant: {value}")

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_unique_object, parse_constant=bad_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read strict JSON input: {exc}") from exc


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise VerificationError(f"{label} must be a non-empty trimmed string")
    return value


def _utc(value: Any, label: str) -> datetime:
    """Accept one unambiguous UTC-second timestamp form; reject offsets/fractions."""
    text = _text(value, label)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", text):
        raise VerificationError(f"{label} must use YYYY-MM-DDTHH:MM:SSZ")
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise VerificationError(f"{label} is not a valid UTC timestamp") from exc


def _b64url_decode(value: Any, label: str, expected_length: int) -> bytes:
    text = _text(value, label)
    if not B64URL.fullmatch(text) or "=" in text or len(text) % 4 == 1:
        raise VerificationError(f"{label} must be unpadded base64url")
    try:
        decoded = base64.urlsafe_b64decode(text + "=" * ((4 - len(text) % 4) % 4))
    except (binascii.Error, ValueError) as exc:
        raise VerificationError(f"{label} is malformed base64url") from exc
    canonical = base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=")
    if canonical != text or len(decoded) != expected_length:
        raise VerificationError(f"{label} has non-canonical encoding or wrong length")
    return decoded


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX_256.fullmatch(value):
        raise VerificationError(f"{label} must be lowercase SHA-256 hex")
    return value


def _context(context: Any) -> dict[str, Any]:
    if not isinstance(context, dict):
        raise VerificationError("expected context must be an object")
    required = {
        "campaign_id", "subject_id", "attempt_id", "receipt_nonce", "scenario_id", "scenario_revision",
        "scenario_digest", "evidence_digest", "role",
    }
    allowed = required | {"proposal_id", "proposal_digest"}
    if required - context.keys() or context.keys() - allowed:
        raise VerificationError("expected context has missing or unknown members")
    _text(context["campaign_id"], "expected campaign_id")
    _text(context["subject_id"], "expected subject_id")
    _text(context["attempt_id"], "expected attempt_id")
    _text(context["scenario_id"], "expected scenario_id")
    if not isinstance(context["scenario_revision"], int) or isinstance(context["scenario_revision"], bool) or not 1 <= context["scenario_revision"] <= (2**53 - 1):
        raise VerificationError("expected scenario_revision must be a safe positive integer")
    _digest(context["scenario_digest"], "expected scenario_digest")
    _digest(context["evidence_digest"], "expected evidence_digest")
    if not isinstance(context["role"], str) or context["role"] not in ROLES:
        raise VerificationError("expected role is unsupported")
    _b64url_decode(context["receipt_nonce"], "expected receipt_nonce", 32)
    if context["role"] == "authorization_decision":
        _text(context.get("proposal_id"), "expected proposal_id")
        _digest(context.get("proposal_digest"), "expected proposal_digest")
    elif "proposal_id" in context or "proposal_digest" in context:
        raise VerificationError("proposal binding is allowed only for authorization_decision")
    return context


def _validate_policy(policy: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(policy, dict) or set(policy) != {"schema_version", "policy_id", "keys"}:
        raise VerificationError("trust policy has invalid top-level shape")
    if policy["schema_version"] != POLICY_VERSION:
        raise VerificationError("unsupported trust-policy schema_version")
    _text(policy["policy_id"], "trust policy policy_id")
    if not isinstance(policy["keys"], list) or not policy["keys"]:
        raise VerificationError("trust policy keys must be a non-empty list")
    by_id: dict[str, dict[str, Any]] = {}
    seen_evaluator_pairs: set[tuple[str, str]] = set()
    seen_public_keys: set[bytes] = set()
    roles_by_evaluator: dict[str, set[str]] = {}
    allowed_key_fields = {
        "key_id", "evaluator_id", "public_key", "roles", "not_before", "not_after", "revoked_at"
    }
    for index, key in enumerate(policy["keys"]):
        label = f"trust policy keys[{index}]"
        if not isinstance(key, dict) or set(key) != allowed_key_fields:
            raise VerificationError(f"{label} has invalid shape")
        key_id = _text(key["key_id"], f"{label}.key_id")
        evaluator_id = _text(key["evaluator_id"], f"{label}.evaluator_id")
        if key_id in by_id:
            raise VerificationError(f"duplicate key_id in trust policy: {key_id}")
        if not isinstance(key["roles"], list) or not key["roles"]:
            raise VerificationError(f"{label}.roles must be a non-empty list")
        if any(not isinstance(role, str) or role not in ROLES for role in key["roles"]):
            raise VerificationError(f"{label}.roles contains unsupported role")
        if len(set(key["roles"])) != len(key["roles"]):
            raise VerificationError(f"{label}.roles contains duplicates")
        pair = (evaluator_id, key_id)
        if pair in seen_evaluator_pairs:
            raise VerificationError(f"duplicate evaluator/key binding: {pair}")
        seen_evaluator_pairs.add(pair)
        public_key_bytes = _b64url_decode(key["public_key"], f"{label}.public_key", 32)
        if public_key_bytes in seen_public_keys:
            raise VerificationError("duplicate public key material in trust policy")
        seen_public_keys.add(public_key_bytes)
        roles_by_evaluator.setdefault(evaluator_id, set()).update(key["roles"])
        not_before = _utc(key["not_before"], f"{label}.not_before")
        not_after = _utc(key["not_after"], f"{label}.not_after")
        if not_before >= not_after:
            raise VerificationError(f"{label} key validity interval is empty or reversed")
        if key["revoked_at"] is not None:
            _utc(key["revoked_at"], f"{label}.revoked_at")
        by_id[key_id] = key

    verdict_roles = {"functional_verdict", "security_verdict"}
    for evaluator_id, roles in roles_by_evaluator.items():
        if verdict_roles <= roles:
            raise VerificationError(
                f"evaluator identity {evaluator_id!r} is assigned both independent verdict roles"
            )
        if "authorization_decision" in roles and roles & verdict_roles:
            raise VerificationError(
                f"evaluator identity {evaluator_id!r} cannot both evaluate and authorize"
            )
    return by_id


def _validate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_FIELDS:
        raise VerificationError("receipt payload has missing or unknown members")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise VerificationError("unsupported receipt schema_version")
    if payload["algorithm"] != "Ed25519":
        raise VerificationError("only Ed25519 signatures are accepted")
    if not isinstance(payload["receipt_id"], str) or not RECEIPT_ID.fullmatch(payload["receipt_id"]):
        raise VerificationError("receipt_id must be 32 lowercase hexadecimal characters")
    for field in ("key_id", "evaluator_id", "campaign_id", "subject_id", "attempt_id", "scenario_id"):
        _text(payload[field], f"payload.{field}")
    if not isinstance(payload["role"], str) or payload["role"] not in ROLES:
        raise VerificationError("payload.role is unsupported")
    if not isinstance(payload["status"], str) or payload["status"] not in STATUSES:
        raise VerificationError("payload.status is unsupported")
    if not isinstance(payload["scenario_revision"], int) or isinstance(payload["scenario_revision"], bool) or not 1 <= payload["scenario_revision"] <= (2**53 - 1):
        raise VerificationError("scenario_revision must be a safe positive integer")
    _digest(payload["scenario_digest"], "payload.scenario_digest")
    _digest(payload["evidence_digest"], "payload.evidence_digest")
    _b64url_decode(payload["receipt_nonce"], "payload.receipt_nonce", 32)
    issued_at = _utc(payload["issued_at"], "payload.issued_at")
    expires_at = _utc(payload["expires_at"], "payload.expires_at")
    if issued_at >= expires_at:
        raise VerificationError("receipt validity interval is empty or reversed")
    proposal_id, proposal_digest = payload["proposal_id"], payload["proposal_digest"]
    if payload["role"] == "authorization_decision":
        _text(proposal_id, "payload.proposal_id")
        _digest(proposal_digest, "payload.proposal_digest")
    elif proposal_id is not None or proposal_digest is not None:
        raise VerificationError("proposal binding is allowed only for authorization_decision")
    return payload


def canonical_signed_bytes(payload: dict[str, Any]) -> bytes:
    """Return domain-separated RFC 8785 canonical bytes to be signed/verified."""
    try:
        canonical = rfc8785.dumps(payload)
    except Exception as exc:
        raise VerificationError(f"payload cannot be RFC 8785 canonicalized: {exc}") from exc
    return DOMAIN_SEPARATOR + canonical


class ReplayLedger:
    """Single-host durable replay fence; protect the DB path and its host.

    Distributed deployments must use a shared store with equivalent atomic
    uniqueness semantics. A caller must supply a freshly issued one-time nonce.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def consume(self, payload: dict[str, Any], now: datetime) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise VerificationError("replay ledger path must not be a symlink")
        if not self.path.exists():
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            except FileExistsError:
                # Another verifier initialized the same protected ledger concurrently.
                pass
            else:
                os.close(fd)
        try:
            if self.path.is_symlink() or not self.path.is_file():
                raise VerificationError("replay ledger path must be a regular file, not a symlink")
            os.chmod(self.path, 0o600)
            with sqlite3.connect(self.path, timeout=10, isolation_level=None) as db:
                db.execute("PRAGMA busy_timeout=10000")
                db.execute(
                    """CREATE TABLE IF NOT EXISTS consumed_receipts (
                        receipt_id TEXT PRIMARY KEY,
                        receipt_nonce TEXT NOT NULL UNIQUE,
                        campaign_id TEXT NOT NULL,
                        consumed_at TEXT NOT NULL
                    )"""
                )
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "INSERT INTO consumed_receipts(receipt_id, receipt_nonce, campaign_id, consumed_at) VALUES (?, ?, ?, ?)",
                    (
                        payload["receipt_id"], payload["receipt_nonce"],
                        payload["campaign_id"], now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    ),
                )
                db.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            # The connection context manager rolls back the failed insert before
            # control reaches this handler.
            raise VerificationError("replay rejected: receipt_id or receipt_nonce was already consumed") from exc


def verify_receipt(
    receipt: Any,
    trust_policy: Any,
    expected_context: Any,
    *,
    now: datetime | None = None,
    replay_ledger: ReplayLedger | None = None,
) -> dict[str, Any]:
    """Verify envelope, trust-policy role, exact expected context, signature, and replay.

    A passing result only authenticates a receipt under the supplied policy. It is
    deliberately not an operational authorization or a claim that underlying events
    or actions were safe.
    """
    if now is None:
        now = datetime.now(timezone.utc).replace(microsecond=0)
    if now.tzinfo is None:
        raise VerificationError("verification time must be timezone-aware")
    now = now.astimezone(timezone.utc)
    if replay_ledger is None:
        raise VerificationError("persistent replay ledger is required for receipt acceptance")
    if not isinstance(receipt, dict) or set(receipt) != {"payload", "signature"}:
        raise VerificationError("receipt envelope must contain exactly payload and signature")
    payload = _validate_payload(receipt["payload"])
    expected = _context(expected_context)
    keys = _validate_policy(trust_policy)
    try:
        trust_policy_digest = hashlib.sha256(rfc8785.dumps(trust_policy)).hexdigest()
    except Exception as exc:
        raise VerificationError(f"trust policy cannot be RFC 8785 canonicalized: {exc}") from exc

    # Bind every context field from the trusted caller to the signed payload.
    for key, value in expected.items():
        payload_key = "scenario_revision" if key == "scenario_revision" else key
        if payload.get(payload_key) != value:
            raise VerificationError(f"receipt does not match expected context field: {key}")
    key_id = payload["key_id"]
    if key_id not in keys:
        raise VerificationError("receipt key_id is not trusted by the supplied policy")
    key = keys[key_id]
    if key["evaluator_id"] != payload["evaluator_id"]:
        raise VerificationError("receipt evaluator_id does not match trusted key identity")
    if payload["role"] not in key["roles"]:
        raise VerificationError("trusted key is not authorized for receipt role")

    issued_at = _utc(payload["issued_at"], "payload.issued_at")
    expires_at = _utc(payload["expires_at"], "payload.expires_at")
    policy_start = _utc(key["not_before"], "trust key not_before")
    policy_end = _utc(key["not_after"], "trust key not_after")
    if issued_at < policy_start or issued_at >= policy_end:
        raise VerificationError("receipt was issued outside trusted key validity")
    if now < policy_start or now >= policy_end:
        raise VerificationError("trusted key is not currently within its validity interval")
    if now < issued_at:
        raise VerificationError("receipt issue time is in the future")
    if now >= expires_at:
        raise VerificationError("receipt has expired")
    if key["revoked_at"] is not None and now >= _utc(key["revoked_at"], "trust key revoked_at"):
        raise VerificationError("trusted key is revoked")

    public_key_bytes = _b64url_decode(key["public_key"], "trusted public_key", 32)
    signature = _b64url_decode(receipt["signature"], "receipt signature", 64)
    try:
        Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(
            signature, canonical_signed_bytes(payload)
        )
    except InvalidSignature as exc:
        raise VerificationError("Ed25519 signature verification failed") from exc
    except (ValueError, TypeError) as exc:
        raise VerificationError(f"invalid Ed25519 verification input: {exc}") from exc

    # This stateful mutation is mandatory and happens only after every other check succeeds.
    replay_ledger.consume(payload, now)

    return {
        "schema_version": "cyber-crucible-receipt-verification-result.v1",
        "signature_verified": True,
        "trust_policy_id": trust_policy["policy_id"],
        "trust_policy_digest": trust_policy_digest,
        "receipt_id": payload["receipt_id"],
        "key_id": payload["key_id"],
        "evaluator_id": payload["evaluator_id"],
        "role": payload["role"],
        "status": payload["status"],
        "campaign_id": payload["campaign_id"],
        "subject_id": payload["subject_id"],
        "attempt_id": payload["attempt_id"],
        "receipt_nonce": payload["receipt_nonce"],
        "issued_at": payload["issued_at"],
        "expires_at": payload["expires_at"],
        "scenario_id": payload["scenario_id"],
        "scenario_revision": payload["scenario_revision"],
        "scenario_digest": payload["scenario_digest"],
        "evidence_digest": payload["evidence_digest"],
        "proposal_id": payload["proposal_id"],
        "proposal_digest": payload["proposal_digest"],
        "replay_checked": True,
        "non_authorizing": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, help="signed receipt JSON file")
    parser.add_argument("--trust-policy", required=True, help="operator-controlled trust policy JSON file")
    parser.add_argument("--expected-context", required=True, help="trusted expected campaign/run/scenario context JSON file")
    parser.add_argument("--replay-db", required=True, help="protected persistent SQLite replay ledger path")
    args = parser.parse_args(argv)
    try:
        result = verify_receipt(
            load_json(args.receipt),
            load_json(args.trust_policy),
            load_json(args.expected_context),
            replay_ledger=ReplayLedger(args.replay_db),
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except (VerificationError, sqlite3.Error, OSError) as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
