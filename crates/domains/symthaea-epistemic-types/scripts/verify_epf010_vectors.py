#!/usr/bin/env python3
"""Independent EPF-010 retrieval-receipt vector verifier."""
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VECTORS = json.loads((ROOT / "vectors" / "epf-010.json").read_text())
DOMAIN = bytes.fromhex(VECTORS["domain_separator_hex"])
TAGS = {"PostFrontier": 0, "FrontierUnknown": 1, "MissingHistoricalFrontier": 2}


def put_string(out, value):
    raw = value.encode("utf-8")
    out.extend(struct.pack(">I", len(raw)))
    out.extend(raw)


def canonical_bytes(receipt):
    """Mirror Rust's canonical encoding, including enum-tag ordering."""
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

    # Rust canonicalization encodes semantic sets in sorted, deduplicated order.
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


for vector in VECTORS["vectors"]:
    receipt = vector["receipt"]
    encoded = canonical_bytes(receipt)
    assert encoded.hex() == vector["canonical_bytes_hex"], vector["name"]
    assert hashlib.sha256(encoded).hexdigest() == vector["receipt_digest"], vector["name"]

print(f"verified {len(VECTORS['vectors'])} EPF-010 retrieval receipt vector(s)")
