#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent compatibility oracle for fabrication trust/clock extraction.

This freezes existing wire/digest semantics before any move into a generic trust
kernel. It grants no runtime authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct

TRUST_DOMAIN = b"symthaea.fabrication.trust-snapshot-digest.v1\0"
OBS_DOMAIN = b"symthaea.fabrication.clock-observation-digest.v1\0"
WINDOW_DOMAIN = b"symthaea.fabrication.verified-clock-window.v1\0"
CONTINUITY_DOMAIN = b"symthaea.fabrication.clock-continuity-digest.v1\0"

EXPECTED = {
    "trust_snapshot": "609805640e3b8d2e110d577a5637e4139fb7864588f0767b4f0b9cfaae20e633",
    "observation_a_epoch_42": "2825223c11e45e51555244013478b5d83feb601d559ba5fcbb6e103b2e220b0c",
    "observation_b_epoch_42": "a5f898ec3f8e095aea0ac7081eda2b249d94dcc39954556a0e0b1da72a9cd1a8",
    "window_epoch_42": "5b41785dcdf0ea4b14e918e0a77d7d8cfa19b55226afe8e487ee07e72c487229",
    "window_epoch_43": "d82fd355f03b961e2ccc52c886e5974084a2f245fea82ea7514432a25c25ab1b",
    "continuity_42_43": "ed00444d5c2a47380ac8e26aead198ab632d0aef092165a3d6c2a353d7f01d15",
}

EXPECTED_TRUST_JSON = (
    '{"schema_version":"symthaea.fabrication.trust-snapshot.v1",'
    '"sequence":7,"issued_at_unix_s":1000,"expires_at_unix_s":2000,'
    '"keys":['
    '{"algorithm":"Ed25519","key_id":"clock-a","not_before_unix_s":900,'
    '"not_after_unix_s":3000,"status":"Active",'
    '"usages":["ClockAuthority","ClockContinuity"]},'
    '{"algorithm":"MlDsa65","key_id":"clock-b","not_before_unix_s":900,'
    '"not_after_unix_s":3000,"status":"Active",'
    '"usages":["ClockAuthority","ClockContinuity"]}'
    ']}'
)

def compact(value) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()

def digest(domain: bytes, payload: bytes) -> bytes:
    return hashlib.sha256(domain + payload).digest()

def trust_snapshot():
    return {
        "schema_version": "symthaea.fabrication.trust-snapshot.v1",
        "sequence": 7,
        "issued_at_unix_s": 1000,
        "expires_at_unix_s": 2000,
        "keys": [
            {
                "algorithm": "Ed25519",
                "key_id": "clock-a",
                "not_before_unix_s": 900,
                "not_after_unix_s": 3000,
                "status": "Active",
                "usages": ["ClockAuthority", "ClockContinuity"],
            },
            {
                "algorithm": "MlDsa65",
                "key_id": "clock-b",
                "not_before_unix_s": 900,
                "not_after_unix_s": 3000,
                "status": "Active",
                "usages": ["ClockAuthority", "ClockContinuity"],
            },
        ],
    }

def observation(source_id, observed_unix_ms, uncertainty_ms, epoch, algorithm, key_id):
    return [
        "symthaea.fabrication.clock-observation.v1",
        source_id,
        observed_unix_ms,
        uncertainty_ms,
        epoch,
        algorithm,
        key_id,
    ]

def observation_digest(value) -> bytes:
    return digest(OBS_DOMAIN, compact(value))

def verified_window(epoch, observations, trust_digest):
    lowers = [obs[2] - obs[3] for obs in observations]
    uppers = [obs[2] + obs[3] for obs in observations]
    lower = max(lowers)
    upper = min(uppers)
    assert lower <= upper
    consensus = lower + (upper - lower) // 2
    accepted = sorted(observation_digest(obs) for obs in observations)
    h = hashlib.sha256()
    h.update(WINDOW_DOMAIN)
    h.update(struct.pack("<Q", epoch))
    h.update(struct.pack("<Q", lower))
    h.update(struct.pack("<Q", upper))
    h.update(trust_digest)
    for item in accepted:
        h.update(item)
    return {
        "lower": lower,
        "upper": upper,
        "consensus": consensus,
        "digest": h.digest(),
    }

def continuity(previous, successor):
    preimage = {
        "schema_version": "symthaea.fabrication.clock-continuity.v1",
        "previous_evidence_digest": list(previous["digest"]),
        "successor_evidence_digest": list(successor["digest"]),
        "previous_epoch": 42,
        "successor_epoch": 43,
        "forward_gap_ms": max(0, successor["lower"] - previous["upper"]),
        "consensus_jump_ms": successor["consensus"] - previous["consensus"],
        "shared_sources": ["source-a", "source-b"],
        "shared_algorithms": ["Ed25519", "MlDsa65"],
        "continuity_digest": [0] * 32,
    }
    return digest(CONTINUITY_DOMAIN, compact(preimage)), preimage

def self_test():
    trust_bytes = compact(trust_snapshot())
    assert trust_bytes.decode() == EXPECTED_TRUST_JSON
    trust = digest(TRUST_DOMAIN, trust_bytes)
    assert trust.hex() == EXPECTED["trust_snapshot"]

    a42 = observation("source-a", 1_500_000, 100, 42, "Ed25519", "clock-a")
    b42 = observation("source-b", 1_500_040, 120, 42, "MlDsa65", "clock-b")
    assert observation_digest(a42).hex() == EXPECTED["observation_a_epoch_42"]
    assert observation_digest(b42).hex() == EXPECTED["observation_b_epoch_42"]

    first = verified_window(42, [b42, a42], trust)
    reordered = verified_window(42, [a42, b42], trust)
    assert first == reordered
    assert (first["lower"], first["upper"], first["consensus"]) == (
        1_499_920,
        1_500_100,
        1_500_010,
    )
    assert first["digest"].hex() == EXPECTED["window_epoch_42"]

    a43 = observation("source-a", 1_500_500, 100, 43, "Ed25519", "clock-a")
    b43 = observation("source-b", 1_500_540, 120, 43, "MlDsa65", "clock-b")
    second = verified_window(43, [a43, b43], trust)
    assert (second["lower"], second["upper"], second["consensus"]) == (
        1_500_420,
        1_500_600,
        1_500_510,
    )
    assert second["digest"].hex() == EXPECTED["window_epoch_43"]

    continuity_digest, preimage = continuity(first, second)
    assert preimage["forward_gap_ms"] == 320
    assert preimage["consensus_jump_ms"] == 500
    assert continuity_digest.hex() == EXPECTED["continuity_42_43"]
    return EXPECTED

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--vectors", action="store_true")
    args = parser.parse_args()
    vectors = self_test()
    if args.self_test:
        for key in sorted(vectors):
            print(f"ok {key}={vectors[key]}")
    elif args.vectors:
        print(json.dumps(vectors, sort_keys=True, separators=(",", ":")))
    else:
        print(json.dumps({"decision": "SelfTest", "vectors": vectors}, sort_keys=True, separators=(",", ":")))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())