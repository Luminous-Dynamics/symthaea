#!/usr/bin/env python3
"""Compile/validate deterministic fsaverage5 -> Glasser360 mapping artifacts.

Input labels are already-decoded semantic HCP-MMP1 names for exactly 10,242
vertices per hemisphere. This tool intentionally does not parse FreeSurfer
`.annot` files: atlas-byte extraction and semantic mapping are separate trust
boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

VERTICES_PER_HEMISPHERE = 10_242
TOTAL_VERTICES = 20_484
AREA_COUNT_PER_HEMISPHERE = 180
TOTAL_PARCELS = 360

AREA_SCHEMA = "symthaea-hcp-mmp1-area-order-v1"
LABEL_SCHEMA = "symthaea-semantic-surface-labels-v1"
ARTIFACT_TYPE = "FsAverage5GlasserMapV1"
SCHEMA_VERSION = 1

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

_AREA_KEYS = {"schema", "atlas", "hemisphere_area_count", "source", "areas"}
_AREA_SOURCE_KEYS = {"repository", "commit", "blob_sha", "path", "purpose"}
_LABEL_KEYS = {"schema", "space", "hemisphere", "vertex_count", "labels", "source"}
_LABEL_SOURCE_KEYS = {
    "source_id",
    "source_version",
    "source_digest",
    "generator_id",
    "generator_version",
    "terms_reference",
}
_ARTIFACT_KEYS = {
    "schema_version",
    "artifact_type",
    "input_space",
    "output_space",
    "vertices_per_hemisphere",
    "hemisphere_order",
    "canonical_area_table",
    "aggregation",
    "source_inputs",
    "vertex_to_parcel",
    "qualification",
    "content_digest",
}
_QUALIFICATION_KEYS = {
    "assigned_vertices_left",
    "assigned_vertices_right",
    "unassigned_vertices_left",
    "unassigned_vertices_right",
    "parcel_vertex_counts",
    "unknown_labels",
    "cross_hemisphere_violations",
}
_AGGREGATION_V1 = {
    "statistic": "arithmetic_mean",
    "non_finite_policy": "reject",
    "empty_parcel_policy": "reject",
    "unassigned_vertex_policy": "exclude",
}


class QualificationError(ValueError):
    """Input or artifact violates the mapping qualification contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_digest(path.read_bytes())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def exact_keys(obj: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise QualificationError(f"{context}: expected an object")
    unknown = sorted(set(obj) - expected)
    missing = sorted(expected - set(obj))
    if unknown:
        raise QualificationError(f"{context}: unknown fields: {', '.join(unknown)}")
    if missing:
        raise QualificationError(f"{context}: missing fields: {', '.join(missing)}")
    return obj


def nonempty_strings(obj: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    for key in keys:
        if not isinstance(obj.get(key), str) or not obj[key].strip():
            raise QualificationError(f"{context}: {key} must be non-empty")


def validate_digest(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise QualificationError(f"{context}: expected sha256:<64 lowercase hex>")
    return value


def load_area_order(path: Path) -> tuple[list[str], dict[str, Any]]:
    doc = exact_keys(load_json(path), _AREA_KEYS, "area order")
    if doc["schema"] != AREA_SCHEMA:
        raise QualificationError(f"area order: schema must be {AREA_SCHEMA}")
    if doc["atlas"] != "HCP-MMP1.0/Glasser360":
        raise QualificationError("area order: unexpected atlas identity")
    if doc["hemisphere_area_count"] != AREA_COUNT_PER_HEMISPHERE:
        raise QualificationError("area order: hemisphere_area_count must be 180")

    source = exact_keys(doc["source"], _AREA_SOURCE_KEYS, "area order source")
    nonempty_strings(source, tuple(_AREA_SOURCE_KEYS), "area order source")
    if not _GIT_SHA_RE.fullmatch(source["commit"]):
        raise QualificationError("area order source: commit must be a 40-hex Git SHA")
    if not _GIT_SHA_RE.fullmatch(source["blob_sha"]):
        raise QualificationError("area order source: blob_sha must be a 40-hex Git blob SHA")

    areas = doc["areas"]
    if not isinstance(areas, list) or len(areas) != AREA_COUNT_PER_HEMISPHERE:
        raise QualificationError("area order: areas must contain exactly 180 names")
    if any(not isinstance(area, str) or not area for area in areas):
        raise QualificationError("area order: every area name must be a non-empty string")
    if any(area.startswith(("L_", "R_")) for area in areas):
        raise QualificationError("area order: names must be hemisphere-neutral base names")
    if len(set(areas)) != AREA_COUNT_PER_HEMISPHERE:
        raise QualificationError("area order: names must be unique")
    return areas, doc


def load_semantic_labels(path: Path, hemisphere: str) -> dict[str, Any]:
    doc = exact_keys(load_json(path), _LABEL_KEYS, hemisphere)
    if doc["schema"] != LABEL_SCHEMA:
        raise QualificationError(f"{hemisphere}: schema must be {LABEL_SCHEMA}")
    if doc["space"] != "fsaverage5":
        raise QualificationError(f"{hemisphere}: space must be fsaverage5")
    if doc["hemisphere"] != hemisphere:
        raise QualificationError(f"{hemisphere}: hemisphere field must be {hemisphere}")
    if doc["vertex_count"] != VERTICES_PER_HEMISPHERE:
        raise QualificationError(
            f"{hemisphere}: vertex_count must be {VERTICES_PER_HEMISPHERE}"
        )
    labels = doc["labels"]
    if not isinstance(labels, list) or len(labels) != VERTICES_PER_HEMISPHERE:
        raise QualificationError(
            f"{hemisphere}: labels must contain exactly {VERTICES_PER_HEMISPHERE} entries"
        )
    if any(label is not None and not isinstance(label, str) for label in labels):
        raise QualificationError(f"{hemisphere}: labels must be canonical strings or null")

    source = exact_keys(doc["source"], _LABEL_SOURCE_KEYS, f"{hemisphere} source")
    nonempty_strings(
        source,
        (
            "source_id",
            "source_version",
            "generator_id",
            "generator_version",
            "terms_reference",
        ),
        f"{hemisphere} source",
    )
    validate_digest(source["source_digest"], f"{hemisphere} source_digest")
    return doc


def compile_hemisphere(
    labels: list[Any], hemisphere: str, area_to_id: dict[str, int]
) -> tuple[list[int | None], list[int]]:
    prefix = "L_" if hemisphere == "left" else "R_"
    offset = 0 if hemisphere == "left" else AREA_COUNT_PER_HEMISPHERE
    mapping: list[int | None] = []
    counts = [0] * AREA_COUNT_PER_HEMISPHERE

    for vertex, label in enumerate(labels):
        if label is None:
            mapping.append(None)
            continue
        if not label.startswith(prefix):
            raise QualificationError(
                f"{hemisphere}: vertex {vertex} label {label!r} violates "
                f"canonical {prefix} hemisphere prefix"
            )
        local_id = area_to_id.get(label[len(prefix) :])
        if local_id is None:
            raise QualificationError(
                f"{hemisphere}: vertex {vertex} has unknown HCP-MMP1 area {label!r}"
            )
        mapping.append(offset + local_id)
        counts[local_id - 1] += 1
    return mapping, counts


def compile_artifact(lh_path: Path, rh_path: Path, area_order_path: Path) -> dict[str, Any]:
    areas, area_doc = load_area_order(area_order_path)
    left = load_semantic_labels(lh_path, "left")
    right = load_semantic_labels(rh_path, "right")
    area_to_id = {area: index + 1 for index, area in enumerate(areas)}

    left_map, left_counts = compile_hemisphere(left["labels"], "left", area_to_id)
    right_map, right_counts = compile_hemisphere(right["labels"], "right", area_to_id)
    parcel_counts = left_counts + right_counts
    empty = [i + 1 for i, count in enumerate(parcel_counts) if count == 0]
    if empty:
        preview = ", ".join(map(str, empty[:12])) + ("..." if len(empty) > 12 else "")
        raise QualificationError(f"complete parcel coverage required; empty parcels: {preview}")

    assigned_left = sum(left_counts)
    assigned_right = sum(right_counts)
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "input_space": "fsaverage5",
        "output_space": "glasser360",
        "vertices_per_hemisphere": VERTICES_PER_HEMISPHERE,
        "hemisphere_order": "left_then_right",
        "canonical_area_table": {
            "schema": AREA_SCHEMA,
            "file_digest": sha256_file(area_order_path),
            "source": area_doc["source"],
        },
        "aggregation": dict(_AGGREGATION_V1),
        "source_inputs": [
            {
                "hemisphere": "left",
                "semantic_labels_file_digest": sha256_file(lh_path),
                **left["source"],
            },
            {
                "hemisphere": "right",
                "semantic_labels_file_digest": sha256_file(rh_path),
                **right["source"],
            },
        ],
        "vertex_to_parcel": left_map + right_map,
        "qualification": {
            "assigned_vertices_left": assigned_left,
            "assigned_vertices_right": assigned_right,
            "unassigned_vertices_left": VERTICES_PER_HEMISPHERE - assigned_left,
            "unassigned_vertices_right": VERTICES_PER_HEMISPHERE - assigned_right,
            "parcel_vertex_counts": parcel_counts,
            "unknown_labels": [],
            "cross_hemisphere_violations": 0,
        },
    }
    artifact["content_digest"] = sha256_digest(canonical_json_bytes(artifact))
    return artifact


def _validate_sources(
    sources: Any, lh_path: Path | None, rh_path: Path | None
) -> None:
    if not isinstance(sources, list) or len(sources) != 2:
        raise QualificationError("artifact: source_inputs must contain left and right")
    if any(not isinstance(source, dict) for source in sources):
        raise QualificationError("artifact: every source_input must be an object")
    if [source.get("hemisphere") for source in sources] != ["left", "right"]:
        raise QualificationError("artifact: source_inputs must be ordered left, right")

    source_keys = _LABEL_SOURCE_KEYS | {"hemisphere", "semantic_labels_file_digest"}
    for source in sources:
        hemi = source["hemisphere"]
        exact_keys(source, source_keys, f"artifact {hemi} source")
        validate_digest(
            source["semantic_labels_file_digest"],
            f"artifact {hemi} semantic label digest",
        )
        validate_digest(source["source_digest"], f"artifact {hemi} source digest")
        nonempty_strings(
            source,
            (
                "source_id",
                "source_version",
                "generator_id",
                "generator_version",
                "terms_reference",
            ),
            f"artifact {hemi} source",
        )

    if (lh_path is None) != (rh_path is None):
        raise QualificationError("artifact: lh/rh semantic label files must be supplied together")
    if lh_path is None or rh_path is None:
        return

    current = [
        (lh_path, load_semantic_labels(lh_path, "left")),
        (rh_path, load_semantic_labels(rh_path, "right")),
    ]
    for source, (path, doc) in zip(sources, current):
        if source["semantic_labels_file_digest"] != sha256_file(path):
            raise QualificationError(
                f"artifact: {source['hemisphere']} semantic label file digest mismatch"
            )
        observed = {key: source[key] for key in _LABEL_SOURCE_KEYS}
        if observed != doc["source"]:
            raise QualificationError(
                f"artifact: {source['hemisphere']} source metadata mismatch"
            )


def validate_artifact(
    artifact: dict[str, Any],
    area_order_path: Path,
    lh_path: Path | None = None,
    rh_path: Path | None = None,
) -> None:
    artifact = exact_keys(artifact, _ARTIFACT_KEYS, "artifact")
    if artifact["schema_version"] != SCHEMA_VERSION:
        raise QualificationError("artifact: unsupported schema_version")
    if artifact["artifact_type"] != ARTIFACT_TYPE:
        raise QualificationError("artifact: unexpected artifact_type")
    if (artifact["input_space"], artifact["output_space"]) != ("fsaverage5", "glasser360"):
        raise QualificationError("artifact: coordinate spaces do not match v1 contract")
    if artifact["vertices_per_hemisphere"] != VERTICES_PER_HEMISPHERE:
        raise QualificationError("artifact: wrong vertices_per_hemisphere")
    if artifact["hemisphere_order"] != "left_then_right":
        raise QualificationError("artifact: hemisphere_order must be left_then_right")

    area_table = exact_keys(
        artifact["canonical_area_table"], {"schema", "file_digest", "source"}, "canonical_area_table"
    )
    if area_table["schema"] != AREA_SCHEMA:
        raise QualificationError("artifact: wrong canonical area schema")
    validate_digest(area_table["file_digest"], "artifact area table digest")
    if area_table["file_digest"] != sha256_file(area_order_path):
        raise QualificationError("artifact: canonical area table digest mismatch")
    _, area_doc = load_area_order(area_order_path)
    if area_table["source"] != area_doc["source"]:
        raise QualificationError("artifact: canonical area table source metadata mismatch")
    if artifact["aggregation"] != _AGGREGATION_V1:
        raise QualificationError("artifact: aggregation policy does not match v1")

    _validate_sources(artifact["source_inputs"], lh_path, rh_path)

    mapping = artifact["vertex_to_parcel"]
    if not isinstance(mapping, list) or len(mapping) != TOTAL_VERTICES:
        raise QualificationError(
            f"artifact: vertex_to_parcel must have exactly {TOTAL_VERTICES} entries"
        )
    for index, parcel in enumerate(mapping):
        if parcel is None:
            continue
        if isinstance(parcel, bool) or not isinstance(parcel, int) or not 1 <= parcel <= TOTAL_PARCELS:
            raise QualificationError(f"artifact: invalid parcel id at vertex {index}")
        if index < VERTICES_PER_HEMISPHERE and parcel > AREA_COUNT_PER_HEMISPHERE:
            raise QualificationError(f"artifact: left vertex {index} maps to right parcel {parcel}")
        if index >= VERTICES_PER_HEMISPHERE and parcel <= AREA_COUNT_PER_HEMISPHERE:
            raise QualificationError(f"artifact: right vertex {index} maps to left parcel {parcel}")

    observed_counts = [0] * TOTAL_PARCELS
    for parcel in mapping:
        if parcel is not None:
            observed_counts[parcel - 1] += 1
    if any(count == 0 for count in observed_counts):
        raise QualificationError("artifact: all 360 parcels require non-zero coverage")

    qualification = exact_keys(
        artifact["qualification"], _QUALIFICATION_KEYS, "artifact qualification"
    )
    if qualification["parcel_vertex_counts"] != observed_counts:
        raise QualificationError("artifact: parcel vertex census mismatch")
    assigned_left = sum(x is not None for x in mapping[:VERTICES_PER_HEMISPHERE])
    assigned_right = sum(x is not None for x in mapping[VERTICES_PER_HEMISPHERE:])
    census = {
        "assigned_vertices_left": assigned_left,
        "assigned_vertices_right": assigned_right,
        "unassigned_vertices_left": VERTICES_PER_HEMISPHERE - assigned_left,
        "unassigned_vertices_right": VERTICES_PER_HEMISPHERE - assigned_right,
    }
    for key, value in census.items():
        if qualification[key] != value:
            raise QualificationError(f"artifact: qualification census mismatch for {key}")
    if qualification["unknown_labels"] != []:
        raise QualificationError("artifact: unknown_labels must be empty")
    if qualification["cross_hemisphere_violations"] != 0:
        raise QualificationError("artifact: cross_hemisphere_violations must be zero")

    digest = validate_digest(artifact["content_digest"], "artifact content_digest")
    body = dict(artifact)
    del body["content_digest"]
    if digest != sha256_digest(canonical_json_bytes(body)):
        raise QualificationError("artifact: content_digest mismatch")


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    compile_cmd = commands.add_parser("compile")
    compile_cmd.add_argument("--lh-labels", type=Path, required=True)
    compile_cmd.add_argument("--rh-labels", type=Path, required=True)
    compile_cmd.add_argument("--area-order", type=Path, required=True)
    compile_cmd.add_argument("--out", type=Path, required=True)

    validate_cmd = commands.add_parser("validate")
    validate_cmd.add_argument("--artifact", type=Path, required=True)
    validate_cmd.add_argument("--area-order", type=Path, required=True)
    validate_cmd.add_argument("--lh-labels", type=Path, required=True)
    validate_cmd.add_argument("--rh-labels", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "compile":
            artifact = compile_artifact(args.lh_labels, args.rh_labels, args.area_order)
            validate_artifact(artifact, args.area_order, args.lh_labels, args.rh_labels)
            write_artifact(args.out, artifact)
            print(f"compiled {ARTIFACT_TYPE} -> {args.out}")
        else:
            artifact = load_json(args.artifact)
            validate_artifact(artifact, args.area_order, args.lh_labels, args.rh_labels)
            print(f"qualified {ARTIFACT_TYPE}: {artifact['content_digest']}")
        return 0
    except (QualificationError, json.JSONDecodeError, OSError) as exc:
        print(f"qualification failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
