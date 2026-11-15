#!/usr/bin/env python3
"""Independent standard-library reference for SYMPsi target mapping.

TEST ONLY: this file is not imported by the Rust package or any production
runner. It independently checks the canonical transcript and rejection mapper.
"""

from __future__ import annotations

import hashlib
import struct
import unittest
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Context:
    protocol_id: bytes
    campaign_id: bytes
    trial_id: bytes
    chain_hash: bytes
    scheme_id: bytes
    round_number: int
    response_commitment: bytes
    verified_randomness: bytes


CHAIN_HASH = bytes.fromhex(
    "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
)
RANDOMNESS = bytes.fromhex(
    "fe290beca10872ef2fb164d2aa4442de4566183ec51c56ff3cd603d930e54fdd"
)
RESPONSE_COMMITMENT = hashlib.sha256(
    b"sympsi-response-commitment-000042"
).digest()
CONTEXT = Context(
    protocol_id=b"sympsi-aia-01-v0.1",
    campaign_id=b"campaign-0001",
    trial_id=b"trial-000042",
    chain_hash=CHAIN_HASH,
    scheme_id=b"bls-unchained-g1-rfc9380",
    round_number=1000,
    response_commitment=RESPONSE_COMMITMENT,
    verified_randomness=RANDOMNESS,
)
EXPECTED_DIGEST = bytes.fromhex(
    "cd4ed4b3a8ab1da6185ea3a535bd6b87b71862f606a9d03c"
    "c5a388e8b854bde6"
)


def length_prefix(value: bytes) -> bytes:
    if len(value) > 0xFFFFFFFF:
        raise ValueError("field exceeds u32 length")
    return struct.pack(">I", len(value)) + value


def transcript(context: Context, counter: int = 0) -> bytes:
    if not (0 <= context.round_number <= 0xFFFFFFFFFFFFFFFF):
        raise ValueError("round is outside u64")
    if not (0 <= counter <= 0xFFFFFFFF):
        raise ValueError("counter is outside u32")
    return b"".join(
        (
            b"sympsi.aia.target.v1\x00",
            length_prefix(context.protocol_id),
            length_prefix(context.campaign_id),
            length_prefix(context.trial_id),
            length_prefix(context.chain_hash),
            length_prefix(context.scheme_id),
            struct.pack(">Q", context.round_number),
            length_prefix(context.response_commitment),
            length_prefix(context.verified_randomness),
            struct.pack(">I", counter),
        )
    )


def map_digest(digest: bytes, choice_count: int) -> int | None:
    if len(digest) != 32:
        raise ValueError("digest must be exactly 32 bytes")
    if choice_count not in (4, 5):
        raise ValueError("only four or five choices are supported")
    integer = int.from_bytes(digest, "big")
    rejection_threshold = (1 << 256) % choice_count
    if integer < rejection_threshold:
        return None
    return integer % choice_count


def target(context: Context, choice_count: int) -> tuple[int, bytes, int]:
    for counter in range(1 << 32):
        digest = hashlib.sha256(transcript(context, counter)).digest()
        mapped = map_digest(digest, choice_count)
        if mapped is not None:
            return mapped, digest, counter
    raise RuntimeError("rejection counter exhausted")


class TargetMappingReferenceTests(unittest.TestCase):
    def test_known_answer_digest_and_four_five_choice_targets(self) -> None:
        digest = hashlib.sha256(transcript(CONTEXT)).digest()
        self.assertEqual(digest, EXPECTED_DIGEST)
        self.assertEqual(target(CONTEXT, 4), (2, EXPECTED_DIGEST, 0))
        self.assertEqual(target(CONTEXT, 5), (2, EXPECTED_DIGEST, 0))

    def test_rejection_threshold_has_no_modulo_bias(self) -> None:
        zero = bytes(32)
        self.assertEqual(map_digest(zero, 4), 0)
        self.assertIsNone(map_digest(zero, 5))
        self.assertEqual(map_digest(bytes([0]) * 31 + bytes([1]), 5), 1)

    def test_every_bound_field_changes_the_transcript_digest(self) -> None:
        baseline = hashlib.sha256(transcript(CONTEXT)).digest()
        mutations = (
            replace(CONTEXT, protocol_id=b"sympsi-aia-02-v0.1"),
            replace(CONTEXT, campaign_id=b"campaign-0002"),
            replace(CONTEXT, trial_id=b"trial-000043"),
            replace(CONTEXT, chain_hash=bytes([1]) * 32),
            replace(CONTEXT, scheme_id=b"another-scheme"),
            replace(CONTEXT, round_number=1001),
            replace(CONTEXT, response_commitment=bytes([1]) * 32),
            replace(CONTEXT, verified_randomness=bytes([2]) * 32),
        )
        for changed in mutations:
            with self.subTest(context=changed):
                self.assertNotEqual(
                    hashlib.sha256(transcript(changed)).digest(),
                    baseline,
                )

    def test_encoding_is_unambiguous_and_fields_are_fixed_width_prefixed(self) -> None:
        self.assertEqual(length_prefix(b"abc"), b"\x00\x00\x00\x03abc")
        self.assertNotEqual(
            length_prefix(b"a") + length_prefix(b"bc"),
            length_prefix(b"ab") + length_prefix(b"c"),
        )

    def test_unsupported_choice_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            target(CONTEXT, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
