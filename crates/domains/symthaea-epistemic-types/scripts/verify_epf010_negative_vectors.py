#!/usr/bin/env python3
"""Independent EPF-010 adversarial-vector verifier (stdlib only)."""
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VECTORS = json.loads((ROOT / "vectors" / "epf-010-negative.json").read_text())
DOMAIN = bytes.fromhex(
    "6570697374656d69632d72657472696576616c2d726563656970743a763100"
)

TAGS = {"PostFrontier": 0, "FrontierUnknown": 1, "MissingHistoricalFrontier": 2}


def put_string(out, value):
    raw = value.encode("utf-8")
    out.extend(struct.pack(">I", len(raw)))
    out.extend(raw)


def canonical_bytes(receipt):
    out = bytearray(DOMAIN)
    out.append({"Historical": 0, "Live": 1}[receipt["mode"]])
    frontier = receipt.get("frontier_ref")
    if frontier is None:
        out.append(0)
    else:
        out.append(1)
        put_string(out, frontier)
    put_string(out, receipt["query"])
    out.extend(struct.pack(">I", receipt["max_results"]))

    selected = sorted(receipt["selected"])
    if len(selected) != len(set(selected)):
        raise ValueError("DuplicateSelectedIdentity")
    out.extend(struct.pack(">I", len(selected)))
    for identity in selected:
        put_string(out, identity)

    bindings = sorted(tuple(pair) for pair in receipt["selected_representation_digests"])
    if len(bindings) != len(set(bindings)):
        raise ValueError("DuplicateRepresentationBinding")
    if any(not digest for _, digest in bindings):
        raise ValueError("EmptyRepresentationDigest")
    if any(identity not in selected for identity, _ in bindings):
        raise ValueError("UnselectedRepresentationIdentity")
    out.extend(struct.pack(">I", len(bindings)))
    for identity, digest in bindings:
        put_string(out, identity)
        put_string(out, digest)

    excluded = sorted(
        receipt["excluded"],
        key=lambda item: (item["canonical_identity"], item["reason"]),
    )
    exclusion_keys = [(item["canonical_identity"], item["reason"]) for item in excluded]
    if len(exclusion_keys) != len(set(exclusion_keys)):
        raise ValueError("DuplicateExclusion")
    out.extend(struct.pack(">I", len(excluded)))
    for item in excluded:
        put_string(out, item["canonical_identity"])
        out.append(TAGS[item["reason"]])

    families = sorted(set(receipt["provenance_families"]))
    out.extend(struct.pack(">I", len(families)))
    for family in families:
        put_string(out, family)

    profiles = sorted(set(receipt["retrieval_profile_versions"]))
    out.extend(struct.pack(">I", len(profiles)))
    for profile in profiles:
        put_string(out, profile)

    return bytes(out)


def verify(vector):
    receipt = vector["receipt"]
    expected = vector["expected_failure"]

    try:
        encoded = canonical_bytes(receipt)
    except ValueError as exc:
        actual = str(exc)
    else:
        override = receipt.get("canonical_bytes_hex_override")
        if override is not None and encoded.hex() != override:
            actual = "CanonicalBytesMismatch"
        elif hashlib.sha256(encoded).hexdigest() != receipt["receipt_digest"]:
            actual = "DigestMismatch"
        else:
            actual = "Accepted"

    assert actual == expected, (vector["name"], expected, actual)


for vector in VECTORS["vectors"]:
    verify(vector)

print(f"verified {len(VECTORS['vectors'])} EPF-010 negative vector(s)")
