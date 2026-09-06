#!/usr/bin/env python3
"""Compare independently derived FsAverage5GlasserMapV1 artifacts.

The comparator is an FMQ-010 evidence tool, not an independence oracle. It
validates both mapping artifacts, emits exact vertex/hemisphere/parcel
agreement evidence, and can require zero unresolved mapping disagreement.
Whether the two source lineages are genuinely independent remains an external
provenance-review obligation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).with_name("compile_fsaverage5_glasser_map.py")
SPEC = importlib.util.spec_from_file_location("fsavg_compiler", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load fsaverage5 Glasser map compiler")
compiler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compiler)

REPORT_TYPE = "FsAverage5GlasserCrosscheckV1"
REPORT_SCHEMA_VERSION = 1


class CrosscheckError(ValueError):
    """Raised when mapping artifacts cannot form a valid cross-check."""


def load_validated(path: Path, area_order: Path) -> dict[str, Any]:
    artifact = compiler.load_json(path)
    compiler.validate_artifact(artifact, area_order)
    return artifact


def lineage_identity(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "content_digest": artifact["content_digest"],
        "canonical_area_table_digest": artifact["canonical_area_table"]["file_digest"],
        "source_inputs": artifact["source_inputs"],
    }


def source_distinctness(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    a_sources = a["source_inputs"]
    b_sources = b["source_inputs"]
    per_hemi: list[dict[str, Any]] = []
    any_distinct = False
    for a_source, b_source in zip(a_sources, b_sources):
        fields = {
            "semantic_labels_file_digest": a_source["semantic_labels_file_digest"]
            != b_source["semantic_labels_file_digest"],
            "source_id": a_source["source_id"] != b_source["source_id"],
            "source_version": a_source["source_version"] != b_source["source_version"],
            "source_digest": a_source["source_digest"] != b_source["source_digest"],
            "generator_id": a_source["generator_id"] != b_source["generator_id"],
            "generator_version": a_source["generator_version"] != b_source["generator_version"],
        }
        distinct = any(fields.values())
        any_distinct = any_distinct or distinct
        per_hemi.append(
            {
                "hemisphere": a_source["hemisphere"],
                "metadata_differs": distinct,
                "different_fields": sorted(key for key, value in fields.items() if value),
            }
        )
    return {
        "metadata_distinctness_detected": any_distinct,
        "per_hemisphere": per_hemi,
        "independence_status": "requires_external_provenance_review",
    }


def disagreement_record(index: int, a_parcel: int | None, b_parcel: int | None) -> dict[str, Any]:
    if index < compiler.VERTICES_PER_HEMISPHERE:
        hemisphere = "left"
        local_vertex = index
    else:
        hemisphere = "right"
        local_vertex = index - compiler.VERTICES_PER_HEMISPHERE
    kind = (
        "assignment_vs_unassigned"
        if (a_parcel is None) != (b_parcel is None)
        else "parcel_mismatch"
    )
    return {
        "vertex": index,
        "hemisphere": hemisphere,
        "local_vertex": local_vertex,
        "lineage_a_parcel": a_parcel,
        "lineage_b_parcel": b_parcel,
        "kind": kind,
    }


def summarize_slice(
    a_map: list[int | None], b_map: list[int | None], start: int, end: int
) -> dict[str, int]:
    same_assigned = 0
    both_unassigned = 0
    assignment_vs_unassigned = 0
    parcel_mismatch = 0
    for a_parcel, b_parcel in zip(a_map[start:end], b_map[start:end]):
        if a_parcel == b_parcel:
            if a_parcel is None:
                both_unassigned += 1
            else:
                same_assigned += 1
        elif (a_parcel is None) != (b_parcel is None):
            assignment_vs_unassigned += 1
        else:
            parcel_mismatch += 1
    return {
        "vertices": end - start,
        "same_assigned_parcel": same_assigned,
        "both_unassigned": both_unassigned,
        "assignment_vs_unassigned": assignment_vs_unassigned,
        "parcel_mismatch": parcel_mismatch,
        "disagreement_vertices": assignment_vs_unassigned + parcel_mismatch,
    }


def parcel_census(a_map: list[int | None], b_map: list[int | None]) -> list[dict[str, int]]:
    a_sets = [set() for _ in range(compiler.TOTAL_PARCELS)]
    b_sets = [set() for _ in range(compiler.TOTAL_PARCELS)]
    for vertex, parcel in enumerate(a_map):
        if parcel is not None:
            a_sets[parcel - 1].add(vertex)
    for vertex, parcel in enumerate(b_map):
        if parcel is not None:
            b_sets[parcel - 1].add(vertex)

    rows = []
    for parcel_index, (a_vertices, b_vertices) in enumerate(zip(a_sets, b_sets), start=1):
        intersection = len(a_vertices & b_vertices)
        only_a = len(a_vertices - b_vertices)
        only_b = len(b_vertices - a_vertices)
        rows.append(
            {
                "parcel": parcel_index,
                "lineage_a_vertices": len(a_vertices),
                "lineage_b_vertices": len(b_vertices),
                "shared_vertices": intersection,
                "only_lineage_a": only_a,
                "only_lineage_b": only_b,
                "symmetric_difference_vertices": only_a + only_b,
            }
        )
    return rows


def build_report(
    lineage_a: dict[str, Any], lineage_b: dict[str, Any]
) -> dict[str, Any]:
    a_map = lineage_a["vertex_to_parcel"]
    b_map = lineage_b["vertex_to_parcel"]
    if len(a_map) != compiler.TOTAL_VERTICES or len(b_map) != compiler.TOTAL_VERTICES:
        raise CrosscheckError("validated mappings unexpectedly have incompatible lengths")

    disagreements = [
        disagreement_record(index, a_parcel, b_parcel)
        for index, (a_parcel, b_parcel) in enumerate(zip(a_map, b_map))
        if a_parcel != b_parcel
    ]
    left = summarize_slice(a_map, b_map, 0, compiler.VERTICES_PER_HEMISPHERE)
    right = summarize_slice(
        a_map, b_map, compiler.VERTICES_PER_HEMISPHERE, compiler.TOTAL_VERTICES
    )
    total = summarize_slice(a_map, b_map, 0, compiler.TOTAL_VERTICES)

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": REPORT_TYPE,
        "input_space": "fsaverage5",
        "output_space": "glasser360",
        "lineage_a": lineage_identity(lineage_a),
        "lineage_b": lineage_identity(lineage_b),
        "source_distinctness": source_distinctness(lineage_a, lineage_b),
        "summary": total,
        "hemispheres": {"left": left, "right": right},
        "parcels": parcel_census(a_map, b_map),
        "disagreements": disagreements,
        "qualification": {
            "exact_mapping_agreement": len(disagreements) == 0,
            "self_comparison": lineage_a["content_digest"] == lineage_b["content_digest"],
            "independence_established": False,
            "independence_note": "source independence cannot be established from artifact metadata alone",
        },
    }
    report["content_digest"] = compiler.sha256_digest(compiler.canonical_json_bytes(report))
    return report


def _validate_summary(value: Any, context: str, expected_vertices: int) -> dict[str, int]:
    keys = {
        "vertices",
        "same_assigned_parcel",
        "both_unassigned",
        "assignment_vs_unassigned",
        "parcel_mismatch",
        "disagreement_vertices",
    }
    summary = compiler.exact_keys(value, keys, context)
    for key in keys:
        if isinstance(summary[key], bool) or not isinstance(summary[key], int) or summary[key] < 0:
            raise CrosscheckError(f"{context}: {key} must be a non-negative integer")
    if summary["vertices"] != expected_vertices:
        raise CrosscheckError(f"{context}: unexpected vertex count")
    if (
        summary["same_assigned_parcel"]
        + summary["both_unassigned"]
        + summary["assignment_vs_unassigned"]
        + summary["parcel_mismatch"]
        != expected_vertices
    ):
        raise CrosscheckError(f"{context}: category census does not sum to vertices")
    if summary["disagreement_vertices"] != (
        summary["assignment_vs_unassigned"] + summary["parcel_mismatch"]
    ):
        raise CrosscheckError(f"{context}: disagreement census mismatch")
    return summary


def validate_report(report: dict[str, Any]) -> None:
    expected = {
        "schema_version",
        "artifact_type",
        "input_space",
        "output_space",
        "lineage_a",
        "lineage_b",
        "source_distinctness",
        "summary",
        "hemispheres",
        "parcels",
        "disagreements",
        "qualification",
        "content_digest",
    }
    report = compiler.exact_keys(report, expected, "crosscheck report")
    if report["schema_version"] != REPORT_SCHEMA_VERSION or report["artifact_type"] != REPORT_TYPE:
        raise CrosscheckError("crosscheck report: unsupported schema/type")
    if (report["input_space"], report["output_space"]) != ("fsaverage5", "glasser360"):
        raise CrosscheckError("crosscheck report: wrong coordinate spaces")

    lineage_keys = {"content_digest", "canonical_area_table_digest", "source_inputs"}
    lineage_a = compiler.exact_keys(report["lineage_a"], lineage_keys, "lineage_a")
    lineage_b = compiler.exact_keys(report["lineage_b"], lineage_keys, "lineage_b")
    for name, lineage in (("lineage_a", lineage_a), ("lineage_b", lineage_b)):
        compiler.validate_digest(lineage["content_digest"], f"{name} content_digest")
        compiler.validate_digest(
            lineage["canonical_area_table_digest"], f"{name} canonical_area_table_digest"
        )
        if not isinstance(lineage["source_inputs"], list) or len(lineage["source_inputs"]) != 2:
            raise CrosscheckError(f"{name}: expected two source_inputs")
    if lineage_a["canonical_area_table_digest"] != lineage_b["canonical_area_table_digest"]:
        raise CrosscheckError("crosscheck report: lineages use different canonical area tables")

    distinct = compiler.exact_keys(
        report["source_distinctness"],
        {"metadata_distinctness_detected", "per_hemisphere", "independence_status"},
        "source_distinctness",
    )
    if distinct["independence_status"] != "requires_external_provenance_review":
        raise CrosscheckError("source_distinctness: independence status may not be upgraded")
    expected_distinct = source_distinctness(
        {"source_inputs": lineage_a["source_inputs"]},
        {"source_inputs": lineage_b["source_inputs"]},
    )
    if distinct != expected_distinct:
        raise CrosscheckError("source_distinctness: metadata comparison mismatch")

    total = _validate_summary(report["summary"], "summary", compiler.TOTAL_VERTICES)
    hemispheres = compiler.exact_keys(report["hemispheres"], {"left", "right"}, "hemispheres")
    left = _validate_summary(
        hemispheres["left"], "hemispheres.left", compiler.VERTICES_PER_HEMISPHERE
    )
    right = _validate_summary(
        hemispheres["right"], "hemispheres.right", compiler.VERTICES_PER_HEMISPHERE
    )
    for key in (
        "same_assigned_parcel",
        "both_unassigned",
        "assignment_vs_unassigned",
        "parcel_mismatch",
        "disagreement_vertices",
    ):
        if total[key] != left[key] + right[key]:
            raise CrosscheckError(f"crosscheck report: hemisphere totals disagree for {key}")

    parcels = report["parcels"]
    if not isinstance(parcels, list) or len(parcels) != compiler.TOTAL_PARCELS:
        raise CrosscheckError("crosscheck report: expected 360 parcel rows")
    parcel_keys = {
        "parcel",
        "lineage_a_vertices",
        "lineage_b_vertices",
        "shared_vertices",
        "only_lineage_a",
        "only_lineage_b",
        "symmetric_difference_vertices",
    }
    for expected_parcel, row_value in enumerate(parcels, start=1):
        row = compiler.exact_keys(row_value, parcel_keys, f"parcel {expected_parcel}")
        if row["parcel"] != expected_parcel:
            raise CrosscheckError("crosscheck report: parcel rows must be canonical 1..360")
        for key in parcel_keys - {"parcel"}:
            if isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] < 0:
                raise CrosscheckError(f"parcel {expected_parcel}: {key} must be non-negative integer")
        if row["lineage_a_vertices"] != row["shared_vertices"] + row["only_lineage_a"]:
            raise CrosscheckError(f"parcel {expected_parcel}: lineage A census mismatch")
        if row["lineage_b_vertices"] != row["shared_vertices"] + row["only_lineage_b"]:
            raise CrosscheckError(f"parcel {expected_parcel}: lineage B census mismatch")
        if row["symmetric_difference_vertices"] != row["only_lineage_a"] + row["only_lineage_b"]:
            raise CrosscheckError(f"parcel {expected_parcel}: symmetric difference mismatch")

    disagreements = report["disagreements"]
    if not isinstance(disagreements, list):
        raise CrosscheckError("crosscheck report: disagreements must be a list")
    disagreement_keys = {
        "vertex",
        "hemisphere",
        "local_vertex",
        "lineage_a_parcel",
        "lineage_b_parcel",
        "kind",
    }
    previous_vertex = -1
    for value in disagreements:
        row = compiler.exact_keys(value, disagreement_keys, "disagreement")
        vertex = row["vertex"]
        if isinstance(vertex, bool) or not isinstance(vertex, int) or not 0 <= vertex < compiler.TOTAL_VERTICES:
            raise CrosscheckError("disagreement: invalid vertex")
        if vertex <= previous_vertex:
            raise CrosscheckError("disagreement: vertices must be strictly increasing")
        previous_vertex = vertex
        expected_hemi = "left" if vertex < compiler.VERTICES_PER_HEMISPHERE else "right"
        expected_local = vertex if expected_hemi == "left" else vertex - compiler.VERTICES_PER_HEMISPHERE
        if row["hemisphere"] != expected_hemi or row["local_vertex"] != expected_local:
            raise CrosscheckError("disagreement: hemisphere/local vertex mismatch")
        if row["lineage_a_parcel"] == row["lineage_b_parcel"]:
            raise CrosscheckError("disagreement: equal assignments are not disagreements")
        expected_kind = (
            "assignment_vs_unassigned"
            if (row["lineage_a_parcel"] is None) != (row["lineage_b_parcel"] is None)
            else "parcel_mismatch"
        )
        if row["kind"] != expected_kind:
            raise CrosscheckError("disagreement: kind mismatch")
    if total["disagreement_vertices"] != len(disagreements):
        raise CrosscheckError("crosscheck report: disagreement list/census mismatch")

    qualification = compiler.exact_keys(
        report["qualification"],
        {
            "exact_mapping_agreement",
            "self_comparison",
            "independence_established",
            "independence_note",
        },
        "qualification",
    )
    if qualification["exact_mapping_agreement"] is not (len(disagreements) == 0):
        raise CrosscheckError("qualification: exact_mapping_agreement mismatch")
    if qualification["self_comparison"] is not (
        lineage_a["content_digest"] == lineage_b["content_digest"]
    ):
        raise CrosscheckError("qualification: self_comparison mismatch")
    if qualification["independence_established"] is not False:
        raise CrosscheckError("qualification: comparator cannot establish source independence")
    if qualification["independence_note"] != (
        "source independence cannot be established from artifact metadata alone"
    ):
        raise CrosscheckError("qualification: independence note mismatch")

    digest = compiler.validate_digest(report["content_digest"], "crosscheck content_digest")
    body = dict(report)
    del body["content_digest"]
    if digest != compiler.sha256_digest(compiler.canonical_json_bytes(body)):
        raise CrosscheckError("crosscheck report: content_digest mismatch")


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage-a", type=Path, required=True)
    parser.add_argument("--lineage-b", type=Path, required=True)
    parser.add_argument("--area-order", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="return non-zero if any vertex mapping disagreement exists",
    )
    parser.add_argument(
        "--reject-self-comparison",
        action="store_true",
        help="return non-zero when the two artifact content digests are identical",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        lineage_a = load_validated(args.lineage_a, args.area_order)
        lineage_b = load_validated(args.lineage_b, args.area_order)
        report = build_report(lineage_a, lineage_b)
        validate_report(report)
        write_report(args.out, report)
        print(f"wrote {REPORT_TYPE} -> {args.out}")
        print(report["content_digest"])
        if args.reject_self_comparison and report["qualification"]["self_comparison"]:
            print("crosscheck failed: self-comparison is not independent evidence", file=sys.stderr)
            return 3
        if args.require_exact and not report["qualification"]["exact_mapping_agreement"]:
            print(
                f"crosscheck failed: {report['summary']['disagreement_vertices']} mapping disagreements",
                file=sys.stderr,
            )
            return 4
        return 0
    except (compiler.QualificationError, CrosscheckError, json.JSONDecodeError, OSError) as exc:
        print(f"crosscheck failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
