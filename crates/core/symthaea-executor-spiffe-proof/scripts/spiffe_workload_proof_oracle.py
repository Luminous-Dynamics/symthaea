#!/usr/bin/env python3
"""Independent stdlib oracle for EXEC-ID SPIFFE workload proof D1A.

This script imports no Symthaea production code and performs no X.509, SPIFFE,
or signature verification. It reconstructs only the canonical evidence bytes for
the frozen fixture.
"""

from __future__ import annotations

import hashlib
import struct

SCHEMA = 1
PROFILE_X509_SVID_CHALLENGE_POP = 0
CLAIM_DOMAIN = b"symthaea.executor.spiffe.workload-proof.claim.v1\0"
CHAIN_DOMAIN = b"symthaea.executor.spiffe.workload-proof.chain.v1\0"
ENVELOPE_DOMAIN = b"symthaea.executor.spiffe.workload-proof.envelope.v1\0"

CHALLENGE = bytes.fromhex(
    "85d9076bfaaf5137f1801ccb7dce4ccd0bcfc35ff87ca1bcc139e13ccf4cda41"
)
SPIFFE_ID = b"spiffe://example.org/workload/api"
CHAIN = [bytes.fromhex("3003010203"), bytes.fromhex("30020405")]
SIGNATURE = bytes.fromhex("aabbccddeeff")

EXPECTED_LEAF = "3f6e6cc25161138d9067ffad9e9eaaf57bbb82fba3cf83c4ec37efa96cf318ba"
EXPECTED_STATEMENT = (
    "73796d74686165612e6578656375746f722e7370696666652e776f726b6c6f61642d70726f6f662e636c61696d2e763100"
    "00010000"
    "85d9076bfaaf5137f1801ccb7dce4ccd0bcfc35ff87ca1bcc139e13ccf4cda41"
    "00000021"
    "7370696666653a2f2f6578616d706c652e6f72672f776f726b6c6f61642f617069"
    "3f6e6cc25161138d9067ffad9e9eaaf57bbb82fba3cf83c4ec37efa96cf318ba"
)
EXPECTED_CLAIM = "feb5411e3bba766511fde2c16307faa55093e7301019ed881e41e843ada55fd2"
EXPECTED_CHAIN = "0d8997a95e42b40c07e938ae2bb648c6b8df1ca4b9beb4368c1f7c8ef00471b9"
EXPECTED_PROOF = "309391c2040694ddb59bf8a85573f489fe1396b134612791a59eb7ffe6c24163"


def u16(value: int) -> bytes:
    return struct.pack(">H", value)


def u32(value: int) -> bytes:
    return struct.pack(">I", value)


def lp32(value: bytes) -> bytes:
    return u32(len(value)) + value


def sha256(value: bytes) -> bytes:
    return hashlib.sha256(value).digest()


def leaf_digest(chain: list[bytes]) -> bytes:
    return sha256(chain[0])


def claim_statement(
    challenge: bytes = CHALLENGE,
    spiffe_id: bytes = SPIFFE_ID,
    leaf: bytes | None = None,
) -> bytes:
    if leaf is None:
        leaf = leaf_digest(CHAIN)
    return b"".join(
        (
            CLAIM_DOMAIN,
            u16(SCHEMA),
            u16(PROFILE_X509_SVID_CHALLENGE_POP),
            challenge,
            lp32(spiffe_id),
            leaf,
        )
    )


def claim_digest(**kwargs: object) -> bytes:
    return sha256(claim_statement(**kwargs))


def chain_digest(chain: list[bytes] = CHAIN) -> bytes:
    transcript = bytearray(CHAIN_DOMAIN)
    transcript.extend(u16(SCHEMA))
    transcript.extend(u32(len(chain)))
    for certificate in chain:
        transcript.extend(lp32(certificate))
    return sha256(bytes(transcript))


def proof_digest(
    claim: bytes | None = None,
    chain: bytes | None = None,
    signature: bytes = SIGNATURE,
) -> bytes:
    if claim is None:
        claim = claim_digest()
    if chain is None:
        chain = chain_digest()
    return sha256(
        b"".join(
            (
                ENVELOPE_DOMAIN,
                u16(SCHEMA),
                claim,
                chain,
                lp32(signature),
            )
        )
    )


def require(label: str, actual: bytes, expected: str) -> None:
    actual_hex = actual.hex()
    if actual_hex != expected:
        raise SystemExit(f"{label} mismatch: {actual_hex} != {expected}")
    print(f"{label:12s} {actual_hex}")


def main() -> None:
    leaf = leaf_digest(CHAIN)
    statement = claim_statement(leaf=leaf)
    require("leaf", leaf, EXPECTED_LEAF)
    require("statement", statement, EXPECTED_STATEMENT)
    require("claim", sha256(statement), EXPECTED_CLAIM)
    require("chain", chain_digest(), EXPECTED_CHAIN)
    require("proof", proof_digest(), EXPECTED_PROOF)

    # Challenge, SPIFFE-ID, and leaf substitution must change the signed claim.
    assert claim_digest(challenge=bytes([0x77]) * 32) != claim_digest()
    assert claim_digest(spiffe_id=b"spiffe://example.org/workload/other") != claim_digest()
    assert claim_digest(leaf=bytes([0x88]) * 32) != claim_digest()

    # Changing only an intermediate path certificate must not change the signed
    # claim, but it must change chain/envelope audit identity.
    changed_path = [CHAIN[0], CHAIN[1] + b"\x88"]
    assert claim_digest() == claim_digest(leaf=leaf_digest(changed_path))
    assert chain_digest(changed_path) != chain_digest()
    assert proof_digest(chain=chain_digest(changed_path)) != proof_digest()

    # Changing only the proof signature leaves claim/path identity intact but
    # changes the ordinary evidence-envelope identity.
    changed_signature = bytes([SIGNATURE[0] ^ 1]) + SIGNATURE[1:]
    assert proof_digest(signature=changed_signature) != proof_digest()


if __name__ == "__main__":
    main()
