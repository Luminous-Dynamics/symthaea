#!/usr/bin/env python3
"""Independent EPF-010 verifier for adversarial receipt vectors (stdlib only)."""
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
    """Mirror Rust's deterministic encoding; structural checks happen after digest."""
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

    # Rust canonicalization treats selected identities as a set.
    selected = sorted(set(receipt["selected"]))
    out.extend(struct.pack(">I", len(selected)))
    for identity in selected:
        put_string(out, identity)

    bindings = sorted(tuple(pair) for pair in receipt["selected_representation_digests"])
    out.extend(struct.pack(">I", len(bindings)))
    for identity, digest in bindings:
        put_string(out, identity)
        put_string(out, digest)

    projection_bindings = sorted(
        tuple(pair) for pair in receipt["selected_projection_identity_digests"]
    )
    out.extend(struct.pack(">I", len(projection_bindings)))
    for identity, digest in projection_bindings:
        put_string(out, identity)
        put_string(out, digest)

    excluded = sorted(
        receipt["excluded"],
        key=lambda item: (item["canonical_identity"], TAGS[item["reason"]]),
    )
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


def semantic_failure(receipt):
    """Mirror MemoryRetrievalReceipt::verify's post-digest rejection order."""
    selected = receipt["selected"]
    if receipt["mode"] == "Historical":
        frontier = receipt.get("frontier_ref")
        if frontier is None:
            return "InvalidRequest(MissingHistoricalFrontier)"
        if frontier == "":
            return "InvalidRequest(EmptyHistoricalFrontier)"
        if not frontier.strip():
            return "InvalidRequest(WhitespaceOnlyHistoricalFrontier)"

    if len(selected) != len(set(selected)):
        return "DuplicateSelectedIdentity"

    selected_set = set(selected)
    projection_bindings = sorted(
        tuple(pair) for pair in receipt["selected_projection_identity_digests"]
    )
    if any(not digest for _, digest in projection_bindings):
        return "EmptyProjectionIdentityDigest"
    if len(projection_bindings) != len(set(projection_bindings)):
        return "DuplicateProjectionIdentityBinding"
    if any(identity not in selected_set for identity, _ in projection_bindings):
        return "UnselectedProjectionIdentity"
    if any(
        not any(bound == identity for bound, _ in projection_bindings)
        for identity in selected
    ):
        return "MissingProjectionIdentityBinding"

    bindings = sorted(tuple(pair) for pair in receipt["selected_representation_digests"])
    if any(not digest for _, digest in bindings):
        return "EmptyRepresentationDigest"
    if len(bindings) != len(set(bindings)):
        return "DuplicateRepresentationBinding"
    if any(identity not in selected_set for identity, _ in bindings):
        return "UnselectedRepresentationIdentity"
    if any(not any(bound == identity for bound, _ in bindings) for identity in selected):
        return "MissingRepresentationBinding"
    if any(
        sum(bound == identity for bound, _ in bindings)
        != sum(bound == identity for bound, _ in projection_bindings)
        for identity in selected
    ):
        return "BindingCountMismatch"

    excluded = sorted(
        receipt["excluded"],
        key=lambda item: (
            item["canonical_identity"],
            TAGS[item["reason"]],
        ),
    )
    if any(left == right for left, right in zip(excluded, excluded[1:])):
        return "DuplicateExclusion"

    profiles = receipt["retrieval_profile_versions"]
    if any(not profile for profile in profiles):
        return "EmptyRetrievalProfileVersion"
    if any(not profile.strip() for profile in profiles):
        return "WhitespaceOnlyRetrievalProfileVersion"
    if len(profiles) != len(set(profiles)):
        return "DuplicateRetrievalProfileVersion"

    return "Accepted"


def verify(vector):
    receipt = vector["receipt"]
    expected = vector["expected_failure"]

    try:
        encoded = canonical_bytes(receipt)
    except (KeyError, TypeError, ValueError, struct.error) as exc:
        actual = f"CanonicalizationError({type(exc).__name__})"
    else:
        override = receipt.get("canonical_bytes_hex_override")
        if override is not None and encoded.hex() != override:
            actual = "CanonicalBytesMismatch"
        elif not receipt.get("receipt_digest") or hashlib.sha256(encoded).hexdigest() != receipt["receipt_digest"]:
            actual = "DigestMismatch"
        else:
            actual = semantic_failure(receipt)

    assert actual == expected, (vector["name"], expected, actual)


for vector in VECTORS["vectors"]:
    verify(vector)

print(f"verified {len(VECTORS['vectors'])} EPF-010 negative vector(s)")
