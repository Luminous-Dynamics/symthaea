#!/usr/bin/env python3
"""Compile and validate a deterministic fsaverage5 -> Glasser360 mapping artifact.

This compiler intentionally does not parse FreeSurfer .annot files. Its inputs are
already-decoded semantic HCP-MMP1 labels for exactly 10,242 vertices per hemisphere.
That keeps binary atlas extraction separate from the scientific mapping authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

VERTICES_PER_HEMISPHERE = 10_242
TOTAL_VERTICES = VERTICES_PER_HEMISPHERE * 2
AREA_COUNT_PER_HEMISPHERE = 180
TOTAL_PARCELS = AREA_COUNT_PER_HEMISPHERE * 2

AREA_SCHEMA = "symthaea-hcp-mmp1-area-order-v1"
LABEL_SCHEMA = "symthaea-semantic-surface-labels-v1"
ARTIFACT_TYPE = "FsAverage5GlasserMapV1"
SCHEMA_VERSION = 1

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class QualificationError(ValueError):
    """Raised when an input cannot satisfy the mapping qualification contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_digest(path.read_bytes())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def reject_unknown_keys(obj: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise QualificationError(f"{context}: unknown fields: {', '.join(unknown)}")


def require_keys(obj: dict[str, Any], required: set[str], context: str) -> None:
    missing = sorted(required - set(obj))
    if missing:
        raise QualificationError(f"{context}: missing fields: {', '.join(missing)}")


def validate_digest(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise QualificationError(f"{context}: expected sha256:<64 lowercase hex>")
    return value


def load_area_order(path: Path) -> tuple[list[str], dict[str, Any]]:
    doc = load_json(path)
    if not isinstance(doc, dict):
        raise QualificationError("area order: top-level JSON must be an object")
    reject_unknown_keys(
        doc,
        {"schema", "atlas", "hemisphere_area_count", "source", "areas"},
        "area order",
    )
    if doc.get("schema") != AREA_SCHEMA:
        raise QualificationError(f"area order: schema must be {AREA_SCHEMA}")
    if doc.get("atlas") != "HCP-MMP1.0/Glasser360":
        raise QualificationError("area order: unexpected atlas identity")
    if doc.get("hemisphere_area_count") != AREA_COUNT_PER_HEMISPHERE:
        raise QualificationError("area order: hemisphere_area_count must be 180")

    source = doc.get("source")
    if not isinstance(source, dict):
        raise QualificationError("area order: source must be an object")
    area_source_keys = {"repository", "commit", "path", "purpose"}
    reject_unknown_keys(source, area_source_keys, "area order source")
    require_keys(source, area_source_keys, "area order source")
    for key in ("repository", "commit", "path", "purpose"):
        if not isinstance(source.get(key), str) or not source[key].strip():
            raise QualificationError(f"area order source: {key} must be non-empty")

    areas = doc.get("areas")
    if not isinstance(areas, list) or len(areas) != AREA_COUNT_PER_HEMISPHERE:
        raise QualificationError("area order: areas must contain exactly 180 names")
    if any(not isinstance(area, str) or not area for area in areas):
        raise QualificationError("area order: every area name must be a non-empty string")
    if any(area.startswith(("L_", "R_")) for area in areas):
        raise QualificationError("area order: names must be hemisphere-neutral base names")
    if len(set(areas)) != AREA_COUNT_PER_HEMISPHERE:
        raise QualificationError("area order: names must be unique")
    return areas, doc


def load_semantic_labels(path: Path, expected_hemisphere: str) -> dict[str, Any]:
    doc = load_json(path)
    if not isinstance(doc, dict):
        raise QualificationError(f"{expected_hemisphere}: top-level JSON must be an object")
    reject_unknown_keys(
        doc,
        {
            "schema",
            "space",
            "hemisphere",
            "vertex_count",
            "labels",
            "source",
        },
        expected_hemisphere,
    )
    if doc.get("schema") != LABEL_SCHEMA:
        raise QualificationError(f"{expected_hemisphere}: schema must be {LABEL_SCHEMA}")
    if doc.get("space") != "fsaverage5":
        raise QualificationError(f"{expected_hemisphere}: space must be fsaverage5")
    if doc.get("hemisphere") != expected_hemisphere:
        raise QualificationError(
            f"{expected_hemisphere}: hemisphere field must be {expected_hemisphere}"
        )
    if doc.get("vertex_count") != VERTICES_PER_HEMISPHERE:
        raise QualificationError(
            f"{expected_hemisphere}: vertex_count must be {VERTICES_PER_HEMISPHERE}"
        )
    labels = doc.get("labels")
    if not isinstance(labels, list) or len(labels) != VERTICES_PER_HEMISPHERE:
        raise QualificationError(
            f"{expected_hemisphere}: labels must contain exactly "
            f"{VERTICES_PER_HEMISPHERE} entries"
        )
    if any(label is not None and not isinstance(label, str) for label in labels):
        raise QualificationError(
            f"{expected_hemisphere}: labels must be canonical strings or null"
        )

    source = doc.get("source")
    if not isinstance(source, dict):
        raise QualificationError(f"{expected_hemisphere}: source must be an object")
    surface_source_keys = {
        "source_id",
        "source_version",
        "source_digest",
        "generator_id",
        "generator_version",
        "terms_reference",
    }
    reject_unknown_keys(source, surface_source_keys, f"{expected_hemisphere} source")
    require_keys(source, surface_source_keys, f"{expected_hemisphere} source")
    for key in (
        "source_id",
        "source_version",
        "generator_id",
        "generator_version",
        "terms_reference",
    ):
        if not isinstance(source.get(key), str) or not source[key].strip():
            raise QualificationError(
                f"{expected_hemisphere} source: {key} must be non-empty"
            )
    validate_digest(source.get("source_digest"), f"{expected_hemisphere} source_digest")
    return doc


def compile_hemisphere(
    labels: list[Any],
    hemisphere: str,
    area_to_local_id: dict[str, int],
) -> tuple[list[int | None], list[int]]:
    prefix = "L_" if hemisphere == "left" else "R_"
    offset = 0 if hemisphere == "left" else AREA_COUNT_PER_HEMISPHERE
    output: list[int | None] = []
    counts = [0] * AREA_COUNT_PER_HEMISPHERE

    for vertex, label in enumerate(labels):
        if label is None:
            output.append(None)
            continue
        if not label.startswith(prefix):
            raise QualificationError(
                f"{hemisphere}: vertex {vertex} label {label!r} violates "
                f"canonical {prefix} hemisphere prefix"
            )
        base = label[len(prefix) :]
        local_id = area_to_local_id.get(base)
        if local_id is None:
            raise QualificationError(
                f"{hemisphere}: vertex {vertex} has unknown HCP-MMP1 area {label!r}"
            )
        parcel_id = offset + local_id
        output.append(parcel_id)
        counts[local_id - 1] += 1

    return output, counts


def compile_artifact(
    lh_path: Path,
    rh_path: Path,
    area_order_path: Path,
) -> dict[str, Any]:
    areas, area_doc = load_area_order(area_order_path)
    left = load_semantic_labels(lh_path, "left")
    right = load_semantic_labels(rh_path, "right")
    area_to_local_id = {area: idx + 1 for idx, area in enumerate(areas)}

    left_map, left_counts = compile_hemisphere(
        left["labels"], "left", area_to_local_id
    )
    right_map, right_counts = compile_hemisphere(
        right["labels"], "right", area_to_local_id
    )
    parcel_counts = left_counts + right_counts
    empty = [idx + 1 for idx, count in enumerate(parcel_counts) if count == 0]
    if empty:
        preview = ", ".join(map(str, empty[:12]))
        suffix = "..." if len(empty) > 12 else ""
        raise QualificationError(
            f"complete parcel coverage required; empty parcels: {preview}{suffix}"
        )

    body: dict[str, Any] = {
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
        "aggregation": {
            "statistic": "arithmetic_mean",
            "non_finite_policy": "reject",
            "empty_parcel_policy": "reject",
            "unassigned_vertex_policy": "exclude",
        },
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
            "assigned_vertices_left": sum(left_counts),
            "assigned_vertices_right": sum(right_counts),
            "unassigned_vertices_left": VERTICES_PER_HEMISPHERE - sum(left_counts),
            "unassigned_vertices_right": VERTICES_PER_HEMISPHERE - sum(right_counts),
            "parcel_vertex_counts": parcel_counts,
            "unknown_labels": [],
            "cross_hemisphere_violations": 0,
        },
    }
    body["content_digest"] = sha256_digest(canonical_json_bytes(body))
    return body


def validate_artifact(
    artifact: dict[str, Any],
    area_order_path: Path,
    lh_path: Path | None = None,
    rh_path: Path | None = None,
) -> None:
    if not isinstance(artifact, dict):
        raise QualificationError("artifact: top-level JSON must be an object")
    expected_keys = {
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
    reject_unknown_keys(artifact, expected_keys, "artifact")
    missing = sorted(expected_keys - set(artifact))
    if missing:
        raise QualificationError(f"artifact: missing fields: {', '.join(missing)}")

    if artifact["schema_version"] != SCHEMA_VERSION:
        raise QualificationError("artifact: unsupported schema_version")
    if artifact["artifact_type"] != ARTIFACT_TYPE:
        raise QualificationError("artifact: unexpected artifact_type")
    if artifact["input_space"] != "fsaverage5" or artifact["output_space"] != "glasser360":
        raise QualificationError("artifact: coordinate spaces do not match v1 contract")
    if artifact["vertices_per_hemisphere"] != VERTICES_PER_HEMISPHERE:
        raise QualificationError("artifact: wrong vertices_per_hemisphere")
    if artifact["hemisphere_order"] != "left_then_right":
        raise QualificationError("artifact: hemisphere_order must be left_then_right")

    area_table = artifact["canonical_area_table"]
    if not isinstance(area_table, dict):
        raise QualificationError("artifact: canonical_area_table must be an object")
    area_table_keys = {"schema", "file_digest", "source"}
    reject_unknown_keys(area_table, area_table_keys, "canonical_area_table")
    require_keys(area_table, area_table_keys, "canonical_area_table")
    if area_table.get("schema") != AREA_SCHEMA:
        raise QualificationError("artifact: wrong canonical area schema")
    validate_digest(area_table.get("file_digest"), "artifact area table digest")
    if area_table["file_digest"] != sha256_file(area_order_path):
        raise QualificationError("artifact: canonical area table digest mismatch")
    _, current_area_doc = load_area_order(area_order_path)
    if area_table["source"] != current_area_doc["source"]:
        raise QualificationError("artifact: canonical area table source metadata mismatch")

    aggregation = artifact["aggregation"]
    expected_aggregation = {
        "statistic": "arithmetic_mean",
        "non_finite_policy": "reject",
        "empty_parcel_policy": "reject",
        "unassigned_vertex_policy": "exclude",
    }
    if aggregation != expected_aggregation:
        raise QualificationError("artifact: aggregation policy does not match v1")

    sources = artifact["source_inputs"]
    if not isinstance(sources, list) or len(sources) != 2:
        raise QualificationError("artifact: source_inputs must contain left and right")
    if [source.get("hemisphere") for source in sources] != ["left", "right"]:
        raise QualificationError("artifact: source_inputs must be ordered left, right")
    source_keys = {
        "hemisphere",
        "semantic_labels_file_digest",
        "source_id",
        "source_version",
        "source_digest",
        "generator_id",
        "generator_version",
        "terms_reference",
    }
    for source in sources:
        hemi = source.get("hemisphere")
        reject_unknown_keys(source, source_keys, f"artifact {hemi} source")
        require_keys(source, source_keys, f"artifact {hemi} source")
        validate_digest(
            source.get("semantic_labels_file_digest"),
            f"artifact {hemi} semantic label digest",
        )
        validate_digest(
            source.get("source_digest"),
            f"artifact {hemi} source digest",
        )
        for key in ("source_id", "source_version", "generator_id", "generator_version", "terms_reference"):
            if not isinstance(source.get(key), str) or not source[key].strip():
                raise QualificationError(f"artifact {hemi} source: {key} must be non-empty")

    if (lh_path is None) != (rh_path is None):
        raise QualificationError("artifact: lh/rh semantic label files must be supplied together")
    if lh_path is not None and rh_path is not None:
        current_left = load_semantic_labels(lh_path, "left")
        current_right = load_semantic_labels(rh_path, "right")
        for idx, (path, current) in enumerate(((lh_path, current_left), (rh_path, current_right))):
            source = sources[idx]
            if source["semantic_labels_file_digest"] != sha256_file(path):
                raise QualificationError(
                    f"artifact: {source['hemisphere']} semantic label file digest mismatch"
                )
            expected_source = dict(current["source"])
            observed_source = {key: source[key] for key in expected_source}
            if observed_source != expected_source:
                raise QualificationError(
                    f"artifact: {source['hemisphere']} source metadata mismatch"
                )

    mapping = artifact["vertex_to_parcel"]
    if not isinstance(mapping, list) or len(mapping) != TOTAL_VERTICES:
        raise QualificationError(
            f"artifact: vertex_to_parcel must have exactly {TOTAL_VERTICES} entries"
        )
    for idx, parcel in enumerate(mapping):
        if parcel is None:
            continue
        if not isinstance(parcel, int) or isinstance(parcel, bool) or not 1 <= parcel <= TOTAL_PARCELS:
            raise QualificationError(f"artifact: invalid parcel id at vertex {idx}")
        if idx < VERTICES_PER_HEMISPHERE and parcel > AREA_COUNT_PER_HEMISPHERE:
            raise QualificationError(f"artifact: left vertex {idx} maps to right parcel {parcel}")
        if idx >= VERTICES_PER_HEMISPHERE and parcel <= AREA_COUNT_PER_HEMISPHERE:
            raise QualificationError(f"artifact: right vertex {idx} maps to left parcel {parcel}")

    observed_counts = [0] * TOTAL_PARCELS
    for parcel in mapping:
        if parcel is not None:
            observed_counts[parcel - 1] += 1
    if any(count == 0 for count in observed_counts):
        raise QualificationError("artifact: all 360 parcels require non-zero coverage")

    qualification = artifact["qualification"]
    if not isinstance(qualification, dict):
        raise QualificationError("artifact: qualification must be an object")
    reject_unknown_keys(
        qualification,
        {
            "assigned_vertices_left",
            "assigned_vertices_right",
            "unassigned_vertices_left",
            "unassigned_vertices_right",
            "parcel_vertex_counts",
            "unknown_labels",
            "cross_hemisphere_violations",
        },
        "artifact qualification",
    )
    if qualification["parcel_vertex_counts"] != observed_counts:
        raise QualificationError("artifact: parcel vertex census mismatch")
    assigned_left = sum(1 for parcel in mapping[:VERTICES_PER_HEMISPHERE] if parcel is not None)
    assigned_right = sum(1 for parcel in mapping[VERTICES_PER_HEMISPHERE:] if parcel is not None)
    expected_census = {
        "assigned_vertices_left": assigned_left,
        "assigned_vertices_right": assigned_right,
        "unassigned_vertices_left": VERTICES_PER_HEMISPHERE - assigned_left,
        "unassigned_vertices_right": VERTICES_PER_HEMISPHERE - assigned_right,
    }
    for key, value in expected_census.items():
        if qualification.get(key) != value:
            raise QualificationError(f"artifact: qualification census mismatch for {key}")
    if qualification.get("unknown_labels") != []:
        raise QualificationError("artifact: unknown_labels must be empty")
    if qualification.get("cross_hemisphere_violations") != 0:
        raise QualificationError("artifact: cross_hemisphere_violations must be zero")

    digest = validate_digest(artifact["content_digest"], "artifact content_digest")
    body = dict(artifact)
    del body["content_digest"]
    if digest != sha256_digest(canonical_json_bytes(body)):
        raise QualificationError("artifact: content_digest mismatch")


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            artifact,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def cmd_compile(args: argparse.Namespace) -> int:
    artifact = compile_artifact(args.lh_labels, args.rh_labels, args.area_order)
    validate_artifact(artifact, args.area_order, args.lh_labels, args.rh_labels)
    write_artifact(args.out, artifact)
    print(f"compiled {ARTIFACT_TYPE} -> {args.out}")
    print(artifact["content_digest"])
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    artifact = load_json(args.artifact)
    validate_artifact(artifact, args.area_order, args.lh_labels, args.rh_labels)
    print(f"qualified {ARTIFACT_TYPE}: {artifact['content_digest']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    compile_parser = sub.add_parser("compile", help="compile semantic labels into a mapping artifact")
    compile_parser.add_argument("--lh-labels", type=Path, required=True)
    compile_parser.add_argument("--rh-labels", type=Path, required=True)
    compile_parser.add_argument("--area-order", type=Path, required=True)
    compile_parser.add_argument("--out", type=Path, required=True)
    compile_parser.set_defaults(func=cmd_compile)

    validate_parser = sub.add_parser("validate", help="validate an existing mapping artifact")
    validate_parser.add_argument("--artifact", type=Path, required=True)
    validate_parser.add_argument("--area-order", type=Path, required=True)
    validate_parser.add_argument("--lh-labels", type=Path, required=True)
    validate_parser.add_argument("--rh-labels", type=Path, required=True)
    validate_parser.set_defaults(func=cmd_validate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except (QualificationError, json.JSONDecodeError, OSError) as exc:
        print(f"qualification failed: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
