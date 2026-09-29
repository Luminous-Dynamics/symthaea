#!/usr/bin/env python3
"""Independent EPF-010 retrieval-receipt vector verifier."""
import hashlib, json, struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VECTORS = json.loads((ROOT / "vectors" / "epf-010.json").read_text())
DOMAIN = bytes.fromhex(VECTORS["domain_separator_hex"])

def put_string(value):
    raw = value.encode("utf-8")
    return struct.pack(">I", len(raw)) + raw

def canonical_bytes(receipt):
    out = bytearray(DOMAIN)
    out.append({"Historical": 0, "Live": 1}[receipt["mode"]])
    frontier = receipt.get("frontier_ref")
    if frontier is None:
        out.append(0)
    else:
        out.append(1)
        out.extend(put_string(frontier))
    out.extend(put_string(receipt["query"]))
    out.extend(struct.pack(">I", receipt["max_results"]))
    selected = sorted(receipt["selected"])
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate selected identity")
    out.extend(struct.pack(">I", len(selected)))
    for identity in selected: out.extend(put_string(identity))
    bindings = sorted(tuple(pair) for pair in receipt["selected_representation_digests"])
    if len(bindings) != len(set(bindings)):
        raise ValueError("duplicate representation binding")
    if any(not digest for _, digest in bindings):
        raise ValueError("empty representation digest")
    if any(identity not in selected for identity, _ in bindings):
        raise ValueError("unselected representation identity")
    out.extend(struct.pack(">I", len(bindings)))
    for identity, digest in bindings:
        out.extend(put_string(identity)); out.extend(put_string(digest))
    excluded = sorted(receipt["excluded"], key=lambda x: (x["canonical_identity"], x["reason"]))
    excluded_keys = [(x["canonical_identity"], x["reason"]) for x in excluded]
    if len(excluded_keys) != len(set(excluded_keys)):
        raise ValueError("duplicate exclusion")
    out.extend(struct.pack(">I", len(excluded)))
    tags = {"PostFrontier": 0, "FrontierUnknown": 1, "MissingHistoricalFrontier": 2}
    for item in excluded:
        out.extend(put_string(item["canonical_identity"])); out.append(tags[item["reason"]])
    families = sorted(set(receipt["provenance_families"]))
    out.extend(struct.pack(">I", len(families)))
    for family in families: out.extend(put_string(family))
    profiles = sorted(set(receipt["retrieval_profile_versions"]))
    out.extend(struct.pack(">I", len(profiles)))
    for profile in profiles: out.extend(put_string(profile))
    return bytes(out)

for vector in VECTORS["vectors"]:
    receipt = vector["receipt"]
    encoded = canonical_bytes(receipt)
    assert encoded.hex() == vector["canonical_bytes_hex"], vector["name"]
    assert hashlib.sha256(encoded).hexdigest() == vector["receipt_digest"], vector["name"]
print(f"verified {len(VECTORS['vectors'])} EPF-010 retrieval receipt vector(s)")
