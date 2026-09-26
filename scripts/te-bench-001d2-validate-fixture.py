#!/usr/bin/env python3
import base64
import hashlib
import io
import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path("docs/release/evidence")
TRANSPORT = ROOT / "te-bench-001d1-hostile-workbook-transport-v1.json"
MANIFEST = ROOT / "te-bench-001d1-hostile-workbook-fixture-v1.manifest.json"

EXPECTED_TRANSPORT_SHA256 = "a6372ebe3fd2b3ac91f6893d41c7f704088603c1f872b984cc9e0ac0b2759375"
EXPECTED_TRANSPORT_BLOB = "69d8a1ad8b05c62121ae23ef7a87239832276e3b"
EXPECTED_MANIFEST_SHA256 = "487652d6cf5fec480ce730b216a49be29b60dc7828cabb80fdaafca63c04d0e0"
EXPECTED_MANIFEST_BLOB = "0ecc282e219907501f54e4915455ab85b873f6a9"
EXPECTED_XLSX_SHA256 = "5587aeb93282e28f8811a6a8604a66346d8305859d158b175623c272f064a01b"
EXPECTED_XLSX_BLOB = "7ea4fda45187958ae39b118783bda63fc7cb51e3"
EXPECTED_XLSX_SIZE = 12910
EXPECTED_ENCODED_SIZE = 17216
EXPECTED_SHEETS = [
    "CleanLiteral",
    "MergedHeaders",
    "FormulaCases",
    "UnitTraps",
    "Missingness",
    "UnitContext",
    "Provenance",
    "PropertyKinds",
    "LexicalPrecision",
    "RowOrder",
    "VisibilityIntent",
    "FixtureManifest",
]
EXPECTED_REALIZED = [
    "TE-D1-01", "TE-D1-02", "TE-D1-04", "TE-D1-05", "TE-D1-06",
    "TE-D1-07", "TE-D1-08", "TE-D1-09", "TE-D1-10", "TE-D1-11",
    "TE-D1-12", "TE-D1-13", "TE-D1-14", "TE-D1-15", "TE-D1-16",
    "TE-D1-17", "TE-D1-18", "TE-D1-19", "TE-D1-20", "TE-D1-21",
    "TE-D1-22", "TE-D1-23", "TE-D1-24", "TE-D1-25",
]


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def bind_file(path: Path, sha256: str, blob: str) -> bytes:
    data = path.read_bytes()
    got_sha = hashlib.sha256(data).hexdigest()
    got_blob = git_blob_sha1(data)
    if got_sha != sha256:
        fail(f"{path}: sha256 drift {got_sha}")
    if got_blob != blob:
        fail(f"{path}: git blob drift {got_blob}")
    return data


def main():
    transport_bytes = bind_file(TRANSPORT, EXPECTED_TRANSPORT_SHA256, EXPECTED_TRANSPORT_BLOB)
    manifest_bytes = bind_file(MANIFEST, EXPECTED_MANIFEST_SHA256, EXPECTED_MANIFEST_BLOB)
    transport = json.loads(transport_bytes)
    manifest = json.loads(manifest_bytes)

    if transport.get("schema") != "te-bench-001d1-hostile-workbook-transport-v1":
        fail("transport schema drift")
    if transport.get("authority") != "transport_custody_only_no_parser_or_scientific_authority":
        fail("transport authority drift")
    if manifest.get("schema") != "te-bench-001d1-hostile-workbook-fixture-v1":
        fail("fixture manifest schema drift")
    if manifest.get("authority") != "synthetic_parser_fixture_only_no_scientific_authority":
        fail("fixture authority drift")
    if manifest.get("architecture_head") != "aeec51153668458526fec403b766ce84b781eeb4":
        fail("architecture head drift")

    chunks = transport.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != 6:
        fail("expected exactly six chunks")
    if [c.get("order") for c in chunks] != [1, 2, 3, 4, 5, 6]:
        fail("chunk ordering drift")

    encoded_parts = []
    encoded_total = 0
    for c in chunks:
        path = Path(c["path"])
        data = path.read_bytes()
        if len(data) != c["size_bytes"]:
            fail(f"{path}: size drift {len(data)}")
        if hashlib.sha256(data).hexdigest() != c["sha256"]:
            fail(f"{path}: sha256 drift")
        if git_blob_sha1(data) != c["git_blob_sha1"]:
            fail(f"{path}: git blob drift")
        encoded_parts.append(data)
        encoded_total += len(data)

    if encoded_total != EXPECTED_ENCODED_SIZE or encoded_total != transport.get("encoded_size_bytes"):
        fail(f"encoded size drift {encoded_total}")

    try:
        decoded = base64.b64decode(b"".join(encoded_parts), validate=True)
    except Exception as exc:
        fail(f"strict base64 decode failed: {exc}")

    artifact = transport.get("decoded_artifact", {})
    if len(decoded) != EXPECTED_XLSX_SIZE or len(decoded) != artifact.get("size_bytes"):
        fail(f"decoded size drift {len(decoded)}")
    if hashlib.sha256(decoded).hexdigest() != EXPECTED_XLSX_SHA256:
        fail("decoded XLSX sha256 drift")
    if git_blob_sha1(decoded) != EXPECTED_XLSX_BLOB:
        fail("decoded XLSX git blob drift")
    if artifact.get("sha256") != EXPECTED_XLSX_SHA256 or artifact.get("git_blob_sha1") != EXPECTED_XLSX_BLOB:
        fail("transport decoded-artifact identity drift")

    manifest_artifact = manifest.get("artifact", {})
    if manifest_artifact.get("size_bytes") != EXPECTED_XLSX_SIZE:
        fail("manifest artifact size drift")
    if manifest_artifact.get("sha256") != EXPECTED_XLSX_SHA256:
        fail("manifest artifact sha256 drift")
    if manifest_artifact.get("git_blob_sha1") != EXPECTED_XLSX_BLOB:
        fail("manifest artifact git blob drift")

    try:
        with zipfile.ZipFile(io.BytesIO(decoded), "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                fail(f"ZIP CRC failure in {bad}")
            names = set(zf.namelist())
            for required in {"[Content_Types].xml", "xl/workbook.xml", "xl/_rels/workbook.xml.rels"}:
                if required not in names:
                    fail(f"missing XLSX part {required}")
            workbook_xml = zf.read("xl/workbook.xml")
    except zipfile.BadZipFile as exc:
        fail(f"decoded artifact is not a valid ZIP/XLSX: {exc}")

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(workbook_xml)
    sheets = root.find("m:sheets", ns)
    if sheets is None:
        fail("workbook has no sheets element")
    sheet_nodes = list(sheets)
    actual_sheets = [node.attrib.get("name") for node in sheet_nodes]
    if actual_sheets != EXPECTED_SHEETS or actual_sheets != manifest.get("sheets"):
        fail(f"sheet census/order drift: {actual_sheets}")

    hidden = [
        node.attrib.get("name")
        for node in sheet_nodes
        if node.attrib.get("state", "visible") != "visible"
    ]
    if hidden:
        fail(f"v1 fixture unexpectedly contains hidden sheets: {hidden}")

    realized = manifest.get("realized_cases")
    if realized != EXPECTED_REALIZED or len(realized) != 24 or len(set(realized)) != 24:
        fail("realized hostile-case census drift")

    deferred = manifest.get("deferred_cases")
    if not isinstance(deferred, list) or len(deferred) != 1:
        fail("expected exactly one deferred case")
    d = deferred[0]
    if d.get("id") != "TE-D1-03" or "hidden" not in d.get("feature", "").lower():
        fail("TE-D1-03 hidden-content deferral drift")
    if "do not count" not in d.get("required_future_disposition", ""):
        fail("deferred-case claim ceiling drift")

    if len(manifest.get("claim_ceiling", [])) != 4:
        fail("fixture claim-ceiling census drift")

    print(
        "ok "
        f"chunks=6 realized=24 deferred=1 hidden_sheets=0 "
        f"xlsx_sha256={EXPECTED_XLSX_SHA256} xlsx_git_blob={EXPECTED_XLSX_BLOB}"
    )


if __name__ == "__main__":
    main()
