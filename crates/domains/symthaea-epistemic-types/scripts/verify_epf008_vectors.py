#!/usr/bin/env python3
import hashlib, json, struct
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "vectors" / "epf-008.json"
def lp(value): return struct.pack(">I", len(value)) + value
def canonical(identity):
    return hashlib.sha256(b"memory-canonical:v1\0" + lp(identity.encode("utf-8"))).hexdigest()
def projection(schema, identity, kind, profile, frontier):
    payload = b"memory-projection:v1\0" + struct.pack(">H", schema) + bytes([kind])
    payload += lp(identity.encode("utf-8")) + lp(profile.encode("utf-8"))
    payload += b"\x00" if frontier is None else b"\x01" + lp(frontier.encode("utf-8"))
    return hashlib.sha256(payload).hexdigest()
def main():
    data = json.loads(VECTORS.read_text(encoding="utf-8"))
    for v in data["vectors"]:
        assert canonical(v["canonical_identity"]) == v["canonical_digest"], v["name"]
        p=v["projection"]
        assert projection(p["schema_version"],v["canonical_identity"],p["memory_kind"],p["projection_profile"],p["source_frontier"]) == v["projection_digest"], v["name"]
    print("verified", len(data["vectors"]), "EPF-008 vectors")
if __name__ == "__main__":
    main()
