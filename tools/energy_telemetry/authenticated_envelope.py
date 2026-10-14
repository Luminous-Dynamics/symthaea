"""Authenticated encrypted telemetry envelope prototype.

The outer envelope exposes only protocol version and opaque key IDs. Event
metadata and payload bytes are encrypted together with ChaCha20-Poly1305.
An Ed25519 signature authenticates the canonical outer envelope. Trust/key
distribution, rotation, revocation, transport replay windows and key storage
remain caller responsibilities; this module does not provide them.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import os
import re
import struct
from typing import Mapping

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from tools.energy_telemetry.store_forward import TelemetryEvent, TelemetryQueueError, _utc

ENVELOPE_VERSION = "telemetry-protected/v1"
MAX_ENVELOPE_BYTES = 65_536
MAX_METADATA_BYTES = 8_192
NONCE_BYTES = 12
SIGNATURE_BYTES = 64
KEY_BYTES = 32
KEY_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
EVENT_ID_RE = re.compile(r"^[a-f0-9]{32}$")
DOMAINS = frozenset({"energy", "water", "agriculture", "infrastructure", "spacecraft"})
QUALITIES = frozenset({"validated", "partial", "estimated", "unknown"})


class EnvelopeError(ValueError):
    """Protected envelope is invalid or cannot be authenticated."""


class EnvelopeAuthenticationError(EnvelopeError):
    """Signature or AEAD authentication failed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EnvelopeError("duplicate JSON object key")
        result[key] = value
    return result


def _strict_json(data: bytes) -> object:
    try:
        text = data.decode("ascii")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _constant: (_ for _ in ()).throw(
                EnvelopeError("non-finite JSON number is forbidden")
            ),
        )
    except EnvelopeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise EnvelopeError("invalid canonical JSON") from exc


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(value: object, field: str, *, exact_length: int | None = None) -> bytes:
    if not isinstance(value, str) or len(value) > MAX_ENVELOPE_BYTES * 2:
        raise EnvelopeError(f"{field} must be a bounded base64 string")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise EnvelopeError(f"{field} is not valid base64") from exc
    if _b64(decoded) != value:
        raise EnvelopeError(f"{field} is not canonical base64")
    if exact_length is not None and len(decoded) != exact_length:
        raise EnvelopeError(f"{field} must be {exact_length} bytes")
    return decoded


def _key_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not KEY_ID_RE.fullmatch(value):
        raise EnvelopeError(f"{name} must be an opaque key ID of 1-64 safe characters")
    return value


def _timestamp(value: datetime) -> str:
    _utc(value, "timestamp")
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise EnvelopeError(f"{field} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise EnvelopeError(f"{field} is invalid") from exc
    if _timestamp(parsed) != value:
        raise EnvelopeError(f"{field} is not in canonical timestamp form")
    return parsed


@dataclass(frozen=True)
class TelemetryMetadata:
    """Only metadata needed to interpret an event; never includes identities or coordinates."""

    event_id: str
    domain: str
    source_sequence: int
    observed_at_utc: datetime
    expires_at_utc: datetime
    max_action_age_seconds: int
    priority: int
    quality: str

    def validate(self) -> None:
        # Reuse the central event contract for time, ranges and domain policy.
        try:
            placeholder = TelemetryEvent(
                event_id=self.event_id,
                domain=self.domain,
                source_sequence=self.source_sequence,
                observed_at_utc=self.observed_at_utc,
                expires_at_utc=self.expires_at_utc,
                max_action_age_seconds=self.max_action_age_seconds,
                priority=self.priority,
                quality=self.quality,
                payload_sha256="0" * 64,
                wire_size_bytes=1,
            )
            placeholder.validate()
        except (TypeError, AttributeError, TelemetryQueueError) as exc:
            raise EnvelopeError(f"invalid event metadata: {exc}") from exc

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "event_id": self.event_id,
            "domain": self.domain,
            "source_sequence": self.source_sequence,
            "observed_at_utc": _timestamp(self.observed_at_utc),
            "expires_at_utc": _timestamp(self.expires_at_utc),
            "max_action_age_seconds": self.max_action_age_seconds,
            "priority": self.priority,
            "quality": self.quality,
        }

    @classmethod
    def from_dict(cls, value: object) -> "TelemetryMetadata":
        fields = {
            "event_id", "domain", "source_sequence", "observed_at_utc",
            "expires_at_utc", "max_action_age_seconds", "priority", "quality",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise EnvelopeError("encrypted event metadata has unknown or missing fields")
        metadata = cls(
            event_id=value["event_id"],
            domain=value["domain"],
            source_sequence=value["source_sequence"],
            observed_at_utc=_parse_timestamp(value["observed_at_utc"], "observed_at_utc"),
            expires_at_utc=_parse_timestamp(value["expires_at_utc"], "expires_at_utc"),
            max_action_age_seconds=value["max_action_age_seconds"],
            priority=value["priority"],
            quality=value["quality"],
        )
        metadata.validate()
        return metadata


@dataclass(frozen=True)
class SealedTelemetry:
    """Canonical signed envelope plus queue metadata derived from those exact bytes."""

    event: TelemetryEvent
    envelope_bytes: bytes

    def validate_binding(self) -> None:
        if not isinstance(self.envelope_bytes, bytes):
            raise EnvelopeError("envelope_bytes must be bytes")
        if not 1 <= len(self.envelope_bytes) <= MAX_ENVELOPE_BYTES:
            raise EnvelopeError("protected envelope is outside the supported byte limit")
        digest = hashlib.sha256(self.envelope_bytes).hexdigest()
        if self.event.payload_sha256 != digest:
            raise EnvelopeError("event digest does not match protected envelope bytes")
        if self.event.wire_size_bytes != len(self.envelope_bytes):
            raise EnvelopeError("event wire size does not match protected envelope bytes")
        self.event.validate()


@dataclass(frozen=True)
class OpenedTelemetry:
    """Authenticated event metadata and plaintext after signature and AEAD verification."""

    event: TelemetryEvent
    payload: bytes
    encryption_key_id: str
    signer_key_id: str


def _aad_header(encryption_key_id: str, signer_key_id: str) -> dict[str, str]:
    return {
        "encryption_key_id": encryption_key_id,
        "signer_key_id": signer_key_id,
        "version": ENVELOPE_VERSION,
    }


def _unsigned_outer(
    *, encryption_key_id: str, signer_key_id: str, nonce: bytes, ciphertext: bytes
) -> dict[str, str]:
    return {
        "ciphertext": _b64(ciphertext),
        "encryption_key_id": encryption_key_id,
        "nonce": _b64(nonce),
        "signer_key_id": signer_key_id,
        "version": ENVELOPE_VERSION,
    }


def seal_payload(
    *,
    metadata: TelemetryMetadata,
    payload: bytes,
    encryption_key_id: str,
    encryption_key: bytes,
    signer_key_id: str,
    signing_key: Ed25519PrivateKey,
) -> SealedTelemetry:
    """Encrypt metadata+payload and sign the canonical outer envelope.

    A fresh 96-bit nonce is generated for each call. Production key management
    must enforce a conservative per-key message limit, rotate keys, and retain
    uniqueness/usage state across restarts and backup restoration.
    """
    metadata.validate()
    if not isinstance(payload, bytes):
        raise EnvelopeError("payload must be bytes")
    if not isinstance(encryption_key, bytes) or len(encryption_key) != KEY_BYTES:
        raise EnvelopeError("ChaCha20-Poly1305 key must be exactly 32 bytes")
    encryption_key_id = _key_id(encryption_key_id, "encryption_key_id")
    signer_key_id = _key_id(signer_key_id, "signer_key_id")
    if not isinstance(signing_key, Ed25519PrivateKey):
        raise EnvelopeError("signing_key must be an Ed25519 private key")

    encoded_metadata = _canonical_json(metadata.to_dict())
    if len(encoded_metadata) > MAX_METADATA_BYTES:
        raise EnvelopeError("event metadata is too large")
    # Length-prefix the canonical metadata before appending opaque payload bytes.
    plaintext = struct.pack(">I", len(encoded_metadata)) + encoded_metadata + payload
    nonce = os.urandom(NONCE_BYTES)
    aad_header = _aad_header(encryption_key_id, signer_key_id)
    ciphertext = ChaCha20Poly1305(encryption_key).encrypt(
        nonce, plaintext, _canonical_json(aad_header)
    )
    outer = _unsigned_outer(
        encryption_key_id=encryption_key_id,
        signer_key_id=signer_key_id,
        nonce=nonce,
        ciphertext=ciphertext,
    )
    signature = signing_key.sign(_canonical_json(outer))
    complete = dict(outer)
    complete["signature"] = _b64(signature)
    envelope_bytes = _canonical_json(complete)
    if len(envelope_bytes) > MAX_ENVELOPE_BYTES:
        raise EnvelopeError("protected envelope exceeds maximum wire size")

    event = TelemetryEvent(
        event_id=metadata.event_id,
        domain=metadata.domain,
        source_sequence=metadata.source_sequence,
        observed_at_utc=metadata.observed_at_utc,
        expires_at_utc=metadata.expires_at_utc,
        max_action_age_seconds=metadata.max_action_age_seconds,
        priority=metadata.priority,
        quality=metadata.quality,
        payload_sha256=hashlib.sha256(envelope_bytes).hexdigest(),
        wire_size_bytes=len(envelope_bytes),
    )
    event.validate()
    return SealedTelemetry(event=event, envelope_bytes=envelope_bytes)


def open_envelope(
    envelope_bytes: bytes,
    *,
    encryption_keys: Mapping[str, bytes],
    verification_keys: Mapping[str, Ed25519PublicKey],
) -> OpenedTelemetry:
    """Verify signer before decrypting; reject non-canonical or malformed envelopes."""
    if not isinstance(envelope_bytes, bytes):
        raise EnvelopeError("envelope_bytes must be bytes")
    if not 1 <= len(envelope_bytes) <= MAX_ENVELOPE_BYTES:
        raise EnvelopeError("protected envelope is outside the supported byte limit")
    value = _strict_json(envelope_bytes)
    required = {
        "version", "encryption_key_id", "signer_key_id", "nonce", "ciphertext", "signature"
    }
    if not isinstance(value, dict) or set(value) != required:
        raise EnvelopeError("outer envelope has unknown or missing fields")
    if value["version"] != ENVELOPE_VERSION:
        raise EnvelopeError("unsupported protected envelope version")
    encryption_key_id = _key_id(value["encryption_key_id"], "encryption_key_id")
    signer_key_id = _key_id(value["signer_key_id"], "signer_key_id")
    if _canonical_json(value) != envelope_bytes:
        raise EnvelopeError("outer envelope is not canonical JSON")

    nonce = _unb64(value["nonce"], "nonce", exact_length=NONCE_BYTES)
    ciphertext = _unb64(value["ciphertext"], "ciphertext")
    signature = _unb64(value["signature"], "signature", exact_length=SIGNATURE_BYTES)
    verification_key = verification_keys.get(signer_key_id)
    if not isinstance(verification_key, Ed25519PublicKey):
        raise EnvelopeAuthenticationError("no trusted Ed25519 verification key for signer")
    unsigned = dict(value)
    del unsigned["signature"]
    try:
        verification_key.verify(signature, _canonical_json(unsigned))
    except InvalidSignature as exc:
        raise EnvelopeAuthenticationError("outer envelope signature verification failed") from exc

    encryption_key = encryption_keys.get(encryption_key_id)
    if not isinstance(encryption_key, bytes) or len(encryption_key) != KEY_BYTES:
        raise EnvelopeAuthenticationError("no valid decryption key for envelope key ID")
    aad = _canonical_json(_aad_header(encryption_key_id, signer_key_id))
    try:
        plaintext = ChaCha20Poly1305(encryption_key).decrypt(nonce, ciphertext, aad)
    except InvalidTag as exc:
        raise EnvelopeAuthenticationError("AEAD authentication failed") from exc
    if len(plaintext) < 4:
        raise EnvelopeError("decrypted inner frame is truncated")
    metadata_length = struct.unpack(">I", plaintext[:4])[0]
    if metadata_length > MAX_METADATA_BYTES or len(plaintext) < 4 + metadata_length:
        raise EnvelopeError("decrypted metadata length is invalid")
    metadata_bytes = plaintext[4 : 4 + metadata_length]
    metadata_dict = _strict_json(metadata_bytes)
    if _canonical_json(metadata_dict) != metadata_bytes:
        raise EnvelopeError("decrypted metadata is not canonical JSON")
    metadata = TelemetryMetadata.from_dict(metadata_dict)
    payload = plaintext[4 + metadata_length :]
    event = TelemetryEvent(
        event_id=metadata.event_id,
        domain=metadata.domain,
        source_sequence=metadata.source_sequence,
        observed_at_utc=metadata.observed_at_utc,
        expires_at_utc=metadata.expires_at_utc,
        max_action_age_seconds=metadata.max_action_age_seconds,
        priority=metadata.priority,
        quality=metadata.quality,
        payload_sha256=hashlib.sha256(envelope_bytes).hexdigest(),
        wire_size_bytes=len(envelope_bytes),
    )
    event.validate()
    return OpenedTelemetry(
        event=event,
        payload=payload,
        encryption_key_id=encryption_key_id,
        signer_key_id=signer_key_id,
    )


@dataclass(frozen=True)
class AuthenticatedAck:
    """Receiver-signed acceptance of one exact protected envelope."""

    receiver_key_id: str
    event_id: str
    payload_sha256: str
    acknowledged_at_utc: datetime
    disposition: str = "accepted"


ACK_VERSION = "telemetry-ack/v1"
ACK_ID_RE = EVENT_ID_RE
DIGEST_RE = re.compile(r"^[a-f0-9]{64}$")
MAX_ACK_BYTES = 2_048


def create_ack(
    *,
    receiver_key_id: str,
    event_id: str,
    payload_sha256: str,
    acknowledged_at_utc: datetime,
    signing_key: Ed25519PrivateKey,
) -> bytes:
    """Sign a positive ACK after receiver-side durable acceptance has committed."""
    receiver_key_id = _key_id(receiver_key_id, "receiver_key_id")
    if not isinstance(event_id, str) or not ACK_ID_RE.fullmatch(event_id):
        raise EnvelopeError("ack event_id must be a 32-character lowercase hexadecimal ID")
    if not isinstance(payload_sha256, str) or not DIGEST_RE.fullmatch(payload_sha256):
        raise EnvelopeError("ack payload digest must be a lowercase SHA-256 digest")
    if not isinstance(signing_key, Ed25519PrivateKey):
        raise EnvelopeError("ACK signing_key must be an Ed25519 private key")
    _utc(acknowledged_at_utc, "acknowledged_at_utc")
    unsigned = {
        "acknowledged_at_utc": _timestamp(acknowledged_at_utc),
        "disposition": "accepted",
        "event_id": event_id,
        "payload_sha256": payload_sha256,
        "receiver_key_id": receiver_key_id,
        "version": ACK_VERSION,
    }
    signature = signing_key.sign(_canonical_json(unsigned))
    full = dict(unsigned)
    full["signature"] = _b64(signature)
    encoded = _canonical_json(full)
    if len(encoded) > MAX_ACK_BYTES:
        raise EnvelopeError("authenticated ACK exceeds maximum size")
    return encoded


def verify_ack(
    ack_bytes: bytes,
    *,
    verification_keys: Mapping[str, Ed25519PublicKey],
) -> AuthenticatedAck:
    """Verify a canonical receiver ACK; unknown signers and fields fail closed."""
    if not isinstance(ack_bytes, bytes) or not 1 <= len(ack_bytes) <= MAX_ACK_BYTES:
        raise EnvelopeError("authenticated ACK is outside the supported byte limit")
    value = _strict_json(ack_bytes)
    expected = {
        "version", "receiver_key_id", "event_id", "payload_sha256",
        "acknowledged_at_utc", "disposition", "signature",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise EnvelopeError("ACK has unknown or missing fields")
    if value["version"] != ACK_VERSION:
        raise EnvelopeError("unsupported ACK version")
    if value["disposition"] != "accepted":
        raise EnvelopeError("only a positive accepted ACK can commit delivery")
    receiver_key_id = _key_id(value["receiver_key_id"], "receiver_key_id")
    event_id = value["event_id"]
    if not isinstance(event_id, str) or not ACK_ID_RE.fullmatch(event_id):
        raise EnvelopeError("ack event_id must be a 32-character lowercase hexadecimal ID")
    payload_sha256 = value["payload_sha256"]
    if not isinstance(payload_sha256, str) or not DIGEST_RE.fullmatch(payload_sha256):
        raise EnvelopeError("ack payload digest must be a lowercase SHA-256 digest")
    acknowledged_at_utc = _parse_timestamp(
        value["acknowledged_at_utc"], "acknowledged_at_utc"
    )
    if _canonical_json(value) != ack_bytes:
        raise EnvelopeError("ACK is not canonical JSON")
    signature = _unb64(value["signature"], "signature", exact_length=SIGNATURE_BYTES)
    verification_key = verification_keys.get(receiver_key_id)
    if not isinstance(verification_key, Ed25519PublicKey):
        raise EnvelopeAuthenticationError("no trusted Ed25519 key for ACK receiver")
    unsigned = dict(value)
    del unsigned["signature"]
    try:
        verification_key.verify(signature, _canonical_json(unsigned))
    except InvalidSignature as exc:
        raise EnvelopeAuthenticationError("receiver ACK signature verification failed") from exc
    return AuthenticatedAck(
        receiver_key_id=receiver_key_id,
        event_id=event_id,
        payload_sha256=payload_sha256,
        acknowledged_at_utc=acknowledged_at_utc,
        disposition="accepted",
    )
