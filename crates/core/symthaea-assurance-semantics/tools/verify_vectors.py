#!/usr/bin/env python3
"""Independent ASSURE-002A semantic commitment conformance oracle."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import unicodedata

SCHEMA = "symthaea.assurance.semantic-commitment.v1"
DOMAIN = "symthaea-assurance-semantic-commitment-v1\n"


def field(label: str, value: str) -> str:
    return f"{label} {len(value.encode('utf-8'))}:{value}\n"


def canonical_bytes(vector: dict[str, object]) -> bytes:
    out = DOMAIN
    out += field("schema", SCHEMA)
    out += field("semantic-id", str(vector["semantic_id"]))
    out += field("definition-schema-id", str(vector["definition_schema_id"]))
    out += field(
        "definition-schema-specification-digest",
        str(vector["definition_schema_specification_digest"]),
    )
    out += field("definition-digest", str(vector["definition_digest"]))
    return out.encode("utf-8")


def main() -> int:
    vector_path = pathlib.Path(__file__).resolve().parent.parent / "vectors" / "semantic_commitment_v1.json"
    data = json.loads(vector_path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise SystemExit(f"fixture schema mismatch: {data.get('schema')!r}")

    observed: dict[str, str] = {}
    for vector in data["vectors"]:
        name = str(vector["name"])
        semantic_id = str(vector["semantic_id"])
        schema_id = str(vector["definition_schema_id"])

        semantic_len = len(semantic_id.encode("utf-8"))
        schema_len = len(schema_id.encode("utf-8"))
        if semantic_len != int(vector["expected_semantic_id_utf8_len"]):
            raise SystemExit(
                f"{name}: semantic-id UTF-8 length {semantic_len} != "
                f"{vector['expected_semantic_id_utf8_len']}"
            )
        if schema_len != int(vector["expected_definition_schema_id_utf8_len"]):
            raise SystemExit(
                f"{name}: schema-id UTF-8 length {schema_len} != "
                f"{vector['expected_definition_schema_id_utf8_len']}"
            )

        digest = hashlib.sha256(canonical_bytes(vector)).hexdigest()
        expected = str(vector["expected_digest"])
        if digest != expected:
            raise SystemExit(f"{name}: digest {digest} != {expected}")
        observed[name] = digest
        print(f"{name}: {digest}")

    nfc = next(v for v in data["vectors"] if v["name"] == "unicode-nfc")
    nfd = next(v for v in data["vectors"] if v["name"] == "unicode-nfd")
    nfc_id = str(nfc["semantic_id"])
    nfd_id = str(nfd["semantic_id"])
    if unicodedata.normalize("NFC", nfc_id) != unicodedata.normalize("NFC", nfd_id):
        raise SystemExit("Unicode fixture pair is not canonically equivalent under NFC")
    if nfc_id.encode("utf-8") == nfd_id.encode("utf-8"):
        raise SystemExit("Unicode fixture pair is not byte-distinct")
    if observed["unicode-nfc"] == observed["unicode-nfd"]:
        raise SystemExit("byte-distinct Unicode inputs unexpectedly share a commitment digest")

    print(f"python={sys.version.split()[0]}")
    print("ASSURE-002A independent semantic vectors: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
