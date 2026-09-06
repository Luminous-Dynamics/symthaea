#!/usr/bin/env python3
"""Extract deterministic fsaverage5 HCP-MMP1 semantic labels from fsaverage .annot.

This tool intentionally performs no surface interpolation. FreeSurfer fsaverage5 is
nested in fsaverage, so the canonical fsaverage5 vertex set is the first 10,242
vertices of each fsaverage hemisphere. The input annotation must therefore be a
full fsaverage annotation with exactly 163,842 vertices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FSAVERAGE_VERTICES_PER_HEMISPHERE = 163_842
FSAVERAGE5_VERTICES_PER_HEMISPHERE = 10_242
AREA_COUNT_PER_HEMISPHERE = 180
AREA_SCHEMA = "symthaea-hcp-mmp1-area-order-v1"
OUTPUT_SCHEMA = "symthaea-semantic-surface-labels-v1"
MANIFEST_SCHEMA = "symthaea-hcpmmp1-fsaverage-lineage-v1"
GENERATOR_ID = "symthaea-fsaverage-hcpmmp1-semantic-extractor"
GENERATOR_VERSION = "v1"

_MANIFEST_KEYS = {
    "schema",
    "lineage_id",
    "source_version",
    "atlas",
    "input_space",
    "license",
    "terms_reference",
    "acknowledgement_required",
    "files",
    "provenance",
}
_FILE_KEYS = {"url", "md5", "expected_vertices"}
_PROVENANCE_KEYS = {"article_doi", "source_description", "downsample_rule"}
_AREA_KEYS = {"schema", "atlas", "hemisphere_area_count", "source", "areas"}


class ExtractionError(ValueError):
    """Source bytes or metadata violate the semantic extraction contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def exact_keys(obj: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ExtractionError(f"{context}: expected object")
    unknown = sorted(set(obj) - expected)
    missing = sorted(expected - set(obj))
    if unknown:
        raise ExtractionError(f"{context}: unknown fields: {', '.join(unknown)}")
    if missing:
        raise ExtractionError(f"{context}: missing fields: {', '.join(missing)}")
    return obj


def require_nonempty(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExtractionError(f"{context}: expected non-empty string")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    doc = exact_keys(load_json(path), _MANIFEST_KEYS, "lineage manifest")
    if doc["schema"] != MANIFEST_SCHEMA:
        raise ExtractionError(f"lineage manifest: schema must be {MANIFEST_SCHEMA}")
    if doc["atlas"] != "HCP-MMP1.0/Glasser360":
        raise ExtractionError("lineage manifest: unexpected atlas")
    if doc["input_space"] != "fsaverage":
        raise ExtractionError("lineage manifest: input_space must be fsaverage")
    for key in ("lineage_id", "source_version", "license", "terms_reference"):
        require_nonempty(doc[key], f"lineage manifest {key}")
    if doc["acknowledgement_required"] is not True:
        raise ExtractionError("lineage manifest: acknowledgement_required must be true")
    files = exact_keys(doc["files"], {"left", "right"}, "lineage files")
    for hemi in ("left", "right"):
        entry = exact_keys(files[hemi], _FILE_KEYS, f"lineage files {hemi}")
        require_nonempty(entry["url"], f"lineage files {hemi} url")
        if not isinstance(entry["md5"], str) or len(entry["md5"]) != 32:
            raise ExtractionError(f"lineage files {hemi}: md5 must be 32 hex chars")
        try:
            int(entry["md5"], 16)
        except ValueError as exc:
            raise ExtractionError(f"lineage files {hemi}: invalid md5") from exc
        if entry["expected_vertices"] != FSAVERAGE_VERTICES_PER_HEMISPHERE:
            raise ExtractionError(
                f"lineage files {hemi}: expected_vertices must be "
                f"{FSAVERAGE_VERTICES_PER_HEMISPHERE}"
            )
    prov = exact_keys(doc["provenance"], _PROVENANCE_KEYS, "lineage provenance")
    for key in _PROVENANCE_KEYS:
        require_nonempty(prov[key], f"lineage provenance {key}")
    return doc


def load_area_order(path: Path) -> list[str]:
    doc = exact_keys(load_json(path), _AREA_KEYS, "area order")
    if doc["schema"] != AREA_SCHEMA:
        raise ExtractionError(f"area order: schema must be {AREA_SCHEMA}")
    if doc["atlas"] != "HCP-MMP1.0/Glasser360":
        raise ExtractionError("area order: unexpected atlas")
    if doc["hemisphere_area_count"] != AREA_COUNT_PER_HEMISPHERE:
        raise ExtractionError("area order: hemisphere_area_count must be 180")
    areas = doc["areas"]
    if not isinstance(areas, list) or len(areas) != AREA_COUNT_PER_HEMISPHERE:
        raise ExtractionError("area order: must contain exactly 180 names")
    if len(set(areas)) != AREA_COUNT_PER_HEMISPHERE:
        raise ExtractionError("area order: names must be unique")
    if any(not isinstance(x, str) or not x or x.startswith(("L_", "R_")) for x in areas):
        raise ExtractionError("area order: invalid hemisphere-neutral area name")
    return areas


@dataclass(frozen=True)
class AnnotData:
    vertex_label_ids: list[int]
    id_to_name: dict[int, str]


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def i32(self, context: str) -> int:
        end = self.pos + 4
        if end > len(self.data):
            raise ExtractionError(f"annotation truncated while reading {context}")
        value = struct.unpack(">i", self.data[self.pos:end])[0]
        self.pos = end
        return value

    def raw(self, n: int, context: str) -> bytes:
        if n < 0:
            raise ExtractionError(f"{context}: negative byte count")
        end = self.pos + n
        if end > len(self.data):
            raise ExtractionError(f"annotation truncated while reading {context}")
        value = self.data[self.pos:end]
        self.pos = end
        return value

    def c_string(self, n: int, context: str) -> str:
        if n <= 0 or n > 1_000_000:
            raise ExtractionError(f"{context}: invalid string length {n}")
        raw = self.raw(n, context)
        raw = raw[:-1] if raw.endswith(b"\x00") else raw
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ExtractionError(f"{context}: invalid UTF-8") from exc
        if not text:
            raise ExtractionError(f"{context}: empty string")
        return text


def _packed_id_v1(r: int, g: int, b: int, t: int) -> int:
    return r + (g << 8) + (b << 16) + (t << 24)


def _packed_id_v2(r: int, g: int, b: int, _t: int) -> int:
    return r + (g << 8) + (b << 16)


def _validate_rgba(values: tuple[int, int, int, int], context: str) -> None:
    if any(v < 0 or v > 255 for v in values):
        raise ExtractionError(f"{context}: color components must be 0..255")


def parse_annot_bytes(data: bytes) -> AnnotData:
    rd = _Reader(data)
    n_verts = rd.i32("vertex count")
    if n_verts != FSAVERAGE_VERTICES_PER_HEMISPHERE:
        raise ExtractionError(
            f"annotation vertex count {n_verts}; expected {FSAVERAGE_VERTICES_PER_HEMISPHERE}"
        )

    labels: list[int | None] = [None] * n_verts
    for row in range(n_verts):
        vertex = rd.i32(f"vertex index row {row}")
        label_id = rd.i32(f"annotation id row {row}")
        if vertex < 0 or vertex >= n_verts:
            raise ExtractionError(f"vertex index out of range: {vertex}")
        if labels[vertex] is not None:
            raise ExtractionError(f"duplicate vertex index: {vertex}")
        labels[vertex] = label_id
    if any(v is None for v in labels):
        raise ExtractionError("annotation does not assign every fsaverage vertex")

    ctab_exists = rd.i32("color table existence")
    if ctab_exists == 0:
        raise ExtractionError("annotation color table missing")
    n_entries_raw = rd.i32("color table entry count/version")
    id_to_name: dict[int, str] = {}

    def add_entry(name: str, rgba: tuple[int, int, int, int], version: int) -> None:
        _validate_rgba(rgba, f"color table {name}")
        packed = _packed_id_v1(*rgba) if version == 1 else _packed_id_v2(*rgba)
        previous = id_to_name.get(packed)
        if previous is not None and previous != name:
            raise ExtractionError(
                f"color table id collision {packed}: {previous!r} vs {name!r}"
            )
        id_to_name[packed] = name

    if n_entries_raw > 0:
        n_entries = n_entries_raw
        if n_entries > 100_000:
            raise ExtractionError("color table has implausibly many entries")
        path_len = rd.i32("color table source path length")
        rd.c_string(path_len, "color table source path")
        for i in range(n_entries):
            name_len = rd.i32(f"color table name length {i}")
            name = rd.c_string(name_len, f"color table name {i}")
            rgba = tuple(rd.i32(f"color table rgba {i}") for _ in range(4))
            add_entry(name, rgba, 1)
    else:
        version = -n_entries_raw
        if version != 2:
            raise ExtractionError(f"unsupported color table version: {version}")
        max_entries = rd.i32("color table max entries")
        if max_entries <= 0 or max_entries > 100_000:
            raise ExtractionError("color table max entries invalid")
        path_len = rd.i32("color table source path length")
        rd.c_string(path_len, "color table source path")
        entries_to_read = rd.i32("color table entries to read")
        if entries_to_read < 0 or entries_to_read > max_entries:
            raise ExtractionError("color table entries_to_read invalid")
        seen_structures: set[int] = set()
        for i in range(entries_to_read):
            structure = rd.i32(f"color table structure {i}")
            if structure < 0 or structure >= max_entries:
                raise ExtractionError(f"color table structure out of range: {structure}")
            if structure in seen_structures:
                raise ExtractionError(f"duplicate color table structure: {structure}")
            seen_structures.add(structure)
            name_len = rd.i32(f"color table name length {i}")
            name = rd.c_string(name_len, f"color table name {i}")
            rgba = tuple(rd.i32(f"color table rgba {i}") for _ in range(4))
            add_entry(name, rgba, 2)

    if rd.pos != len(data):
        raise ExtractionError(f"annotation has {len(data) - rd.pos} trailing bytes")

    out_labels = [int(v) for v in labels if v is not None]
    missing_ids = sorted(set(out_labels) - set(id_to_name))
    if missing_ids:
        preview = ", ".join(map(str, missing_ids[:8]))
        raise ExtractionError(f"annotation ids missing from color table: {preview}")
    return AnnotData(out_labels, id_to_name)


def parse_annot(path: Path) -> AnnotData:
    return parse_annot_bytes(path.read_bytes())


def normalize_hcp_label(name: str, hemisphere: str, area_set: set[str]) -> str | None:
    if name == "???":
        return None
    prefix = "L_" if hemisphere == "left" else "R_"
    if not name.startswith(prefix) or not name.endswith("_ROI"):
        raise ExtractionError(
            f"{hemisphere}: unexpected HCP-MMP1 color-table name {name!r}"
        )
    base = name[len(prefix) : -len("_ROI")]
    if base not in area_set:
        raise ExtractionError(f"{hemisphere}: unknown HCP-MMP1 area {name!r}")
    return prefix + base


def extract_semantic_labels(
    annot_path: Path,
    hemisphere: str,
    area_order_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if hemisphere not in ("left", "right"):
        raise ExtractionError("hemisphere must be left or right")
    areas = load_area_order(area_order_path)
    manifest = load_manifest(manifest_path)
    expected = manifest["files"][hemisphere]
    actual_md5 = md5_file(annot_path)
    if actual_md5 != expected["md5"]:
        raise ExtractionError(
            f"{hemisphere}: source MD5 mismatch: {actual_md5} != {expected['md5']}"
        )

    annot = parse_annot(annot_path)
    area_set = set(areas)
    semantic: list[str | None] = []
    for vertex in range(FSAVERAGE5_VERTICES_PER_HEMISPHERE):
        label_id = annot.vertex_label_ids[vertex]
        name = annot.id_to_name[label_id]
        semantic.append(normalize_hcp_label(name, hemisphere, area_set))

    prefix = "L_" if hemisphere == "left" else "R_"
    counts = {area: 0 for area in areas}
    unassigned = 0
    for label in semantic:
        if label is None:
            unassigned += 1
        else:
            counts[label[len(prefix) :]] += 1
    empty = [area for area, count in counts.items() if count == 0]
    if empty:
        raise ExtractionError(
            "fsaverage5 semantic extraction has empty HCP-MMP1 areas: " + ", ".join(empty)
        )

    source_digest = sha256_file(annot_path)
    return {
        "schema": OUTPUT_SCHEMA,
        "space": "fsaverage5",
        "hemisphere": hemisphere,
        "vertex_count": FSAVERAGE5_VERTICES_PER_HEMISPHERE,
        "labels": semantic,
        "source": {
            "source_id": f"{manifest['lineage_id']}:{hemisphere}",
            "source_version": manifest["source_version"],
            "source_digest": source_digest,
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "terms_reference": manifest["terms_reference"],
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="extract fsaverage5 semantic labels")
    extract.add_argument("--annot", type=Path, required=True)
    extract.add_argument("--hemisphere", choices=("left", "right"), required=True)
    extract.add_argument("--area-order", type=Path, required=True)
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--out", type=Path, required=True)

    verify = sub.add_parser("verify-source", help="verify source hash and annotation structure")
    verify.add_argument("--annot", type=Path, required=True)
    verify.add_argument("--hemisphere", choices=("left", "right"), required=True)
    verify.add_argument("--manifest", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "extract":
            doc = extract_semantic_labels(
                args.annot, args.hemisphere, args.area_order, args.manifest
            )
            write_json(args.out, doc)
            return 0
        manifest = load_manifest(args.manifest)
        expected = manifest["files"][args.hemisphere]
        actual = md5_file(args.annot)
        if actual != expected["md5"]:
            raise ExtractionError(
                f"{args.hemisphere}: source MD5 mismatch: {actual} != {expected['md5']}"
            )
        parse_annot(args.annot)
        print(sha256_file(args.annot))
        return 0
    except (ExtractionError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
