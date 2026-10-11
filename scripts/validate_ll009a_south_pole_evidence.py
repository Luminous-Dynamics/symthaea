#!/usr/bin/env python3
"""Validate/materialize the LL-009A South-Pole terrain/site evidence pack.

This is metadata/evidence tooling, not a GIS solver. It makes source lineage,
frames, uncertainty products, no-data policy, candidate-site status, and local
artifact hashes explicit before rover/rail/FLOAT/cable/ballistic trades consume
terrain data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

SCHEMA_VERSION = "ll009a.south-pole-evidence.v1"
ALLOWED_EVIDENCE = {"measured", "derived", "authoritative_metadata", "study_assumption"}
ALLOWED_NODE_STATUS = {"study_hypothesis", "authoritative_selected_site"}


class EvidenceError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n").encode("utf-8")


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_file_entry(entry: dict[str, Any], dataset_id: str) -> None:
    for field in ("filename", "role", "source_reference"):
        if not nonempty(entry.get(field)):
            raise EvidenceError(f"dataset {dataset_id}: file missing {field}")
    if "required_local" in entry and not isinstance(entry["required_local"], bool):
        raise EvidenceError(f"dataset {dataset_id}: required_local must be bool")


def validate_dataset(dataset: dict[str, Any], study_frame: str) -> None:
    dataset_id = str(dataset.get("id", "")).strip()
    if not dataset_id:
        raise EvidenceError("dataset missing id")
    for field in (
        "provider",
        "product",
        "source_reference",
        "frame",
        "projection",
        "horizontal_units",
        "vertical_units",
        "evidence_class",
    ):
        if not nonempty(dataset.get(field)):
            raise EvidenceError(f"dataset {dataset_id}: missing {field}")
    if dataset["evidence_class"] not in ALLOWED_EVIDENCE:
        raise EvidenceError(f"dataset {dataset_id}: invalid evidence_class")
    resolution = dataset.get("resolution_m")
    if resolution is not None and (not isinstance(resolution, (int, float)) or not math.isfinite(resolution) or resolution <= 0):
        raise EvidenceError(f"dataset {dataset_id}: invalid resolution_m")
    files = dataset.get("files")
    if not isinstance(files, list) or not files:
        raise EvidenceError(f"dataset {dataset_id}: files must be non-empty")
    names: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise EvidenceError(f"dataset {dataset_id}: file entry must be object")
        validate_file_entry(entry, dataset_id)
        filename = entry["filename"]
        if filename in names:
            raise EvidenceError(f"dataset {dataset_id}: duplicate file {filename}")
        names.add(filename)

    frame = str(dataset["frame"])
    if frame != study_frame:
        bridge = dataset.get("frame_bridge")
        if not isinstance(bridge, dict):
            raise EvidenceError(
                f"dataset {dataset_id}: frame {frame} differs from study frame {study_frame} without frame_bridge"
            )
        if bridge.get("policy") != "explicit_transform_required":
            raise EvidenceError(f"dataset {dataset_id}: non-study frame must require explicit transform")
        if bridge.get("source_frame") != frame or bridge.get("destination_frame") != study_frame:
            raise EvidenceError(f"dataset {dataset_id}: frame_bridge endpoints do not close")
        if not nonempty(bridge.get("evidence_ref")):
            raise EvidenceError(f"dataset {dataset_id}: frame_bridge missing evidence_ref")

    uncertainty = dataset.get("uncertainty_products", [])
    if not isinstance(uncertainty, list):
        raise EvidenceError(f"dataset {dataset_id}: uncertainty_products must be list")
    file_names = {entry["filename"] for entry in files}
    for name in uncertainty:
        if name not in file_names:
            raise EvidenceError(f"dataset {dataset_id}: uncertainty product {name} not listed in files")


def validate_node(node: dict[str, Any], dataset_ids: set[str]) -> None:
    node_id = str(node.get("id", "")).strip()
    if not node_id or not nonempty(node.get("label")):
        raise EvidenceError("candidate node requires id and label")
    status = node.get("status")
    if status not in ALLOWED_NODE_STATUS:
        raise EvidenceError(f"node {node_id}: invalid status")
    dataset_id = node.get("source_dataset_id")
    if dataset_id not in dataset_ids:
        raise EvidenceError(f"node {node_id}: unknown source_dataset_id")
    if status == "authoritative_selected_site" and not nonempty(node.get("authoritative_selection_ref")):
        raise EvidenceError(f"node {node_id}: authoritative status requires source reference")

    lat = node.get("latitude_deg")
    lon = node.get("longitude_deg")
    if (lat is None) ^ (lon is None):
        raise EvidenceError(f"node {node_id}: latitude/longitude must appear together")
    if lat is not None:
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lat, lon)):
            raise EvidenceError(f"node {node_id}: non-finite coordinates")
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            raise EvidenceError(f"node {node_id}: coordinate range invalid")
        if not nonempty(node.get("coordinate_frame")) or not nonempty(node.get("coordinate_provenance")):
            raise EvidenceError(f"node {node_id}: coordinates require frame + provenance")


def load_and_validate(path: Path) -> dict[str, Any]:
    pack = json.loads(path.read_text(encoding="utf-8"))
    if pack.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError(f"schema_version must be {SCHEMA_VERSION}")
    if not nonempty(pack.get("pack_id")) or not nonempty(pack.get("study_frame")):
        raise EvidenceError("pack_id and study_frame are required")
    if pack.get("no_data_policy") != "fail_closed":
        raise EvidenceError("no_data_policy must be fail_closed")
    datasets = pack.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise EvidenceError("datasets must be non-empty")
    ids: set[str] = set()
    for dataset in datasets:
        if not isinstance(dataset, dict):
            raise EvidenceError("dataset entries must be objects")
        validate_dataset(dataset, pack["study_frame"])
        if dataset["id"] in ids:
            raise EvidenceError(f"duplicate dataset id: {dataset['id']}")
        ids.add(dataset["id"])
    nodes = pack.get("candidate_nodes", [])
    if not isinstance(nodes, list):
        raise EvidenceError("candidate_nodes must be list")
    node_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            raise EvidenceError("candidate node entries must be objects")
        validate_node(node, ids)
        if node["id"] in node_ids:
            raise EvidenceError(f"duplicate candidate node id: {node['id']}")
        node_ids.add(node["id"])
    return pack


def materialize(pack_path: Path, artifact_root: Path) -> dict[str, Any]:
    pack = load_and_validate(pack_path)
    artifacts: list[dict[str, Any]] = []
    for dataset in pack["datasets"]:
        for entry in dataset["files"]:
            local = artifact_root / entry["filename"]
            if not local.is_file():
                if entry.get("required_local", True):
                    raise EvidenceError(f"missing required artifact: {local}")
                continue
            artifacts.append(
                {
                    "dataset_id": dataset["id"],
                    "filename": entry["filename"],
                    "role": entry["role"],
                    "bytes": local.stat().st_size,
                    "sha256": sha256_file(local),
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "pack_id": pack["pack_id"],
        "source_manifest_filename": pack_path.name,
        "source_manifest_sha256": sha256_file(pack_path),
        "study_frame": pack["study_frame"],
        "no_data_policy": pack["no_data_policy"],
        "artifacts": artifacts,
        "candidate_nodes": pack.get("candidate_nodes", []),
    }


def self_test() -> None:
    good_dataset = {
        "id": "synthetic",
        "provider": "test",
        "product": "synthetic",
        "source_reference": "test:synthetic",
        "frame": "SOURCE_FRAME",
        "projection": "synthetic projection",
        "horizontal_units": "m",
        "vertical_units": "m",
        "resolution_m": 5.0,
        "evidence_class": "measured",
        "files": [
            {"filename": "dem.tif", "role": "elevation", "source_reference": "test:dem"},
            {"filename": "err.tif", "role": "elevation_uncertainty", "source_reference": "test:err"},
        ],
        "uncertainty_products": ["err.tif"],
        "frame_bridge": {
            "policy": "explicit_transform_required",
            "source_frame": "SOURCE_FRAME",
            "destination_frame": "STUDY_FRAME",
            "evidence_ref": "test:bridge",
        },
    }
    validate_dataset(good_dataset, "STUDY_FRAME")
    bad = dict(good_dataset)
    bad.pop("frame_bridge")
    try:
        validate_dataset(bad, "STUDY_FRAME")
    except EvidenceError:
        pass
    else:
        raise AssertionError("frame mismatch without explicit bridge must fail")

    validate_node(
        {
            "id": "n1",
            "label": "hypothesis",
            "status": "study_hypothesis",
            "source_dataset_id": "synthetic",
        },
        {"synthetic"},
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        print("LL-009A self-test: PASS")
        return 0
    if args.manifest is None:
        raise EvidenceError("--manifest is required")
    pack = load_and_validate(args.manifest)
    if args.artifact_root is None:
        print(f"valid: {pack['pack_id']}")
        return 0
    result = materialize(args.manifest, args.artifact_root)
    payload = canonical_json_bytes(result)
    if args.output is None:
        sys.stdout.buffer.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
        print(args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvidenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
