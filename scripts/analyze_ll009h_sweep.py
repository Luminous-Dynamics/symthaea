#!/usr/bin/env python3
"""Analyze LL-009G Pareto-front membership across a declared scenario grid.

LL-009H reports only observed switching brackets between adjacent sampled grid
points. It does not interpolate an exact crossover threshold and does not select
a universal lunar transport architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any, Iterable

MANIFEST_SCHEMA = "ll009h.sweep-manifest.v1"
ANALYSIS_SCHEMA = "ll009g.pareto-analysis.v1"
OUTPUT_SCHEMA = "ll009h.scenario-switching-analysis.v1"
AXIS_KINDS = {"numeric", "categorical"}
FRONTS = ("central", "robust")


class SweepError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SweepError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SweepError(f"expected JSON object: {path}")
    return value


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def safe_manifest_path(value: str) -> PurePosixPath:
    if not nonempty(value) or "\\" in value:
        raise SweepError(f"invalid analysis path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SweepError(f"analysis path escapes/ambiguous: {value!r}")
    return path


def resolve_under(root: Path, rel: PurePosixPath) -> Path:
    root_abs = root.resolve()
    candidate = (root_abs / Path(*rel.parts)).resolve()
    try:
        candidate.relative_to(root_abs)
    except ValueError as exc:
        raise SweepError(f"analysis path escapes root: {rel}") from exc
    return candidate


def validate_axes(value: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(value, list) or not value:
        raise SweepError("axes must be a non-empty list")
    axes: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for raw in value:
        if not isinstance(raw, dict):
            raise SweepError("axis entries must be objects")
        name = raw.get("name")
        kind = raw.get("kind")
        unit = raw.get("unit")
        values = raw.get("values")
        if not nonempty(name) or name in by_name:
            raise SweepError(f"axis name missing/duplicate: {name!r}")
        if kind not in AXIS_KINDS:
            raise SweepError(f"axis {name}: invalid kind {kind!r}")
        if not nonempty(unit):
            raise SweepError(f"axis {name}: missing unit")
        if not isinstance(values, list) or len(values) < 2:
            raise SweepError(f"axis {name}: requires at least two ordered values")
        if kind == "numeric":
            if any(not finite_number(v) for v in values):
                raise SweepError(f"axis {name}: numeric values must be finite numbers")
            if any(not values[i] < values[i + 1] for i in range(len(values) - 1)):
                raise SweepError(f"axis {name}: numeric values must be strictly increasing")
        else:
            if any(not nonempty(v) for v in values):
                raise SweepError(f"axis {name}: categorical values must be non-empty strings")
            if len(values) != len(set(values)):
                raise SweepError(f"axis {name}: categorical values must be unique")
        axis = {"name": name, "kind": kind, "unit": unit, "values": values}
        axes.append(axis)
        by_name[name] = axis
    return axes, by_name


def validate_candidate_catalog(value: Any) -> tuple[list[dict[str, str]], set[str]]:
    if not isinstance(value, list) or len(value) < 2:
        raise SweepError("candidate_catalog requires at least two candidates")
    catalog: list[dict[str, str]] = []
    ids: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise SweepError("candidate catalog entries must be objects")
        cid = item.get("candidate_id")
        architecture = item.get("architecture")
        if not nonempty(cid) or not nonempty(architecture):
            raise SweepError("candidate catalog entry requires candidate_id and architecture")
        if cid in ids:
            raise SweepError(f"duplicate candidate_id in catalog: {cid}")
        ids.add(cid)
        catalog.append({"candidate_id": cid, "architecture": architecture})
    return sorted(catalog, key=lambda x: x["candidate_id"]), ids


def coordinate_key(coordinates: dict[str, Any], axes: list[dict[str, Any]]) -> tuple[int, ...]:
    return tuple(axis["values"].index(coordinates[axis["name"]]) for axis in axes)


def validate_coordinates(
    value: Any,
    axes: list[dict[str, Any]],
    axes_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SweepError("point coordinates must be an object")
    expected = {axis["name"] for axis in axes}
    if set(value) != expected:
        raise SweepError(
            f"coordinate axes mismatch missing={sorted(expected - set(value))} "
            f"extra={sorted(set(value) - expected)}"
        )
    normalized: dict[str, Any] = {}
    for axis in axes:
        name = axis["name"]
        coordinate = value[name]
        if coordinate not in axes_by_name[name]["values"]:
            raise SweepError(
                f"coordinate {name}={coordinate!r} is not a declared axis value"
            )
        normalized[name] = coordinate
    return normalized


def validate_analysis(path: Path, declared_sha256: str, candidate_ids: set[str]) -> dict[str, Any]:
    if not path.is_file():
        raise SweepError(f"missing LL-009G analysis: {path}")
    if not valid_sha256(declared_sha256):
        raise SweepError(f"invalid declared analysis SHA-256 for {path}")
    actual_file_sha = sha256_file(path)
    if actual_file_sha != declared_sha256:
        raise SweepError(
            f"analysis file SHA-256 mismatch {path}: expected={declared_sha256} actual={actual_file_sha}"
        )
    analysis = load_json(path)
    if analysis.get("schema_version") != ANALYSIS_SCHEMA or analysis.get("status") != "pass":
        raise SweepError(f"unsupported/non-pass LL-009G analysis: {path}")
    declared_self = analysis.get("analysis_sha256")
    if not valid_sha256(declared_self):
        raise SweepError(f"analysis missing valid self-hash: {path}")
    unhashed = dict(analysis)
    unhashed.pop("analysis_sha256", None)
    recomputed = sha256_bytes(canonical_json_bytes(unhashed))
    if recomputed != declared_self:
        raise SweepError(
            f"analysis self-hash mismatch {path}: expected={declared_self} actual={recomputed}"
        )
    policy_hash = analysis.get("objective_policy_sha256")
    if not valid_sha256(policy_hash):
        raise SweepError(f"analysis missing valid objective policy hash: {path}")
    objectives = analysis.get("objectives")
    units = analysis.get("units")
    vectors = analysis.get("candidate_vectors")
    central = analysis.get("central_front")
    robust = analysis.get("robust_front")
    if not isinstance(objectives, list) or not objectives:
        raise SweepError(f"analysis missing objectives: {path}")
    if not isinstance(units, dict) or not units:
        raise SweepError(f"analysis missing units: {path}")
    if not isinstance(vectors, dict) or set(vectors) != candidate_ids:
        raise SweepError(f"analysis candidate set does not match candidate catalog: {path}")
    for front_name, front in (("central", central), ("robust", robust)):
        if not isinstance(front, list) or any(cid not in candidate_ids for cid in front):
            raise SweepError(f"analysis invalid {front_name}_front: {path}")
        if len(front) != len(set(front)):
            raise SweepError(f"analysis duplicate candidate in {front_name}_front: {path}")
    return {
        "analysis": analysis,
        "file_sha256": actual_file_sha,
        "self_sha256": declared_self,
        "objective_policy_sha256": policy_hash,
        "objectives": objectives,
        "units": units,
        "candidate_ids": sorted(vectors),
        "central_front": sorted(central),
        "robust_front": sorted(robust),
        "study_id": analysis.get("study_id"),
        "accounting_contract_id": analysis.get("accounting_contract_id"),
    }


def adjacent_pairs(
    point_by_index: dict[tuple[int, ...], dict[str, Any]],
    axes: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any], int]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for index, point in sorted(point_by_index.items()):
        for axis_i, axis in enumerate(axes):
            if index[axis_i] + 1 >= len(axis["values"]):
                continue
            neighbor = list(index)
            neighbor[axis_i] += 1
            other = point_by_index.get(tuple(neighbor))
            if other is not None:
                pairs.append((point, other, axis_i))
    return pairs


def membership_switches(
    low_point: dict[str, Any],
    high_point: dict[str, Any],
    axis: dict[str, Any],
    candidate_ids: set[str],
    front_kind: str,
) -> list[dict[str, Any]]:
    low_front = set(low_point[f"{front_kind}_front"])
    high_front = set(high_point[f"{front_kind}_front"])
    switches: list[dict[str, Any]] = []
    for cid in sorted(candidate_ids):
        before = cid in low_front
        after = cid in high_front
        if before == after:
            continue
        switches.append(
            {
                "candidate_id": cid,
                "front": front_kind,
                "event": "enter" if after else "leave",
                "axis": axis["name"],
                "unit": axis["unit"],
                "from_value": low_point["coordinates"][axis["name"]],
                "to_value": high_point["coordinates"][axis["name"]],
                "from_point_id": low_point["point_id"],
                "to_point_id": high_point["point_id"],
                "fixed_coordinates": {
                    name: value
                    for name, value in low_point["coordinates"].items()
                    if name != axis["name"]
                },
                "semantics": "observed_adjacent_grid_bracket_no_interpolation",
            }
        )
    return switches


def analyze_sweep(manifest_path: Path, analysis_root: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise SweepError(f"sweep manifest schema_version must be {MANIFEST_SCHEMA}")
    sweep_name = manifest.get("sweep_name")
    study_family = manifest.get("study_family_id")
    if not nonempty(sweep_name) or not nonempty(study_family):
        raise SweepError("sweep manifest requires sweep_name and study_family_id")
    interpolation_policy = manifest.get("interpolation_policy")
    if interpolation_policy != "forbidden_v1":
        raise SweepError("LL-009H v1 requires interpolation_policy='forbidden_v1'")

    axes, axes_by_name = validate_axes(manifest.get("axes"))
    catalog, candidate_ids = validate_candidate_catalog(manifest.get("candidate_catalog"))
    catalog_hash = sha256_bytes(canonical_json_bytes(catalog))
    if manifest.get("candidate_catalog_sha256") != catalog_hash:
        raise SweepError(
            f"candidate_catalog_sha256 mismatch expected={catalog_hash} "
            f"actual={manifest.get('candidate_catalog_sha256')}"
        )

    raw_points = manifest.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        raise SweepError("sweep manifest requires at least one sampled point")

    normalized_points: list[dict[str, Any]] = []
    point_by_index: dict[tuple[int, ...], dict[str, Any]] = {}
    point_ids: set[str] = set()
    reference_policy: str | None = None
    reference_objectives: bytes | None = None
    reference_units: bytes | None = None

    for raw in raw_points:
        if not isinstance(raw, dict):
            raise SweepError("point entries must be objects")
        point_id = raw.get("point_id")
        if not nonempty(point_id) or point_id in point_ids:
            raise SweepError(f"point_id missing/duplicate: {point_id!r}")
        point_ids.add(point_id)
        coordinates = validate_coordinates(raw.get("coordinates"), axes, axes_by_name)
        index = coordinate_key(coordinates, axes)
        if index in point_by_index:
            raise SweepError(f"duplicate scenario coordinate: {coordinates}")
        rel = safe_manifest_path(str(raw.get("analysis_path", "")))
        source = resolve_under(analysis_root, rel)
        verified = validate_analysis(source, str(raw.get("sha256", "")), candidate_ids)
        policy = verified["objective_policy_sha256"]
        objective_bytes = canonical_json_bytes(verified["objectives"])
        unit_bytes = canonical_json_bytes(verified["units"])
        if reference_policy is None:
            reference_policy = policy
            reference_objectives = objective_bytes
            reference_units = unit_bytes
        else:
            if policy != reference_policy:
                raise SweepError(f"objective policy mismatch at point {point_id}")
            if objective_bytes != reference_objectives:
                raise SweepError(f"objective definition mismatch at point {point_id}")
            if unit_bytes != reference_units:
                raise SweepError(f"objective unit mismatch at point {point_id}")

        point = {
            "point_id": point_id,
            "coordinates": coordinates,
            "analysis_path": rel.as_posix(),
            "analysis_file_sha256": verified["file_sha256"],
            "analysis_sha256": verified["self_sha256"],
            "study_id": verified["study_id"],
            "accounting_contract_id": verified["accounting_contract_id"],
            "central_front": verified["central_front"],
            "robust_front": verified["robust_front"],
        }
        point_by_index[index] = point
        normalized_points.append(point)

    pairs = adjacent_pairs(point_by_index, axes)
    central_switches: list[dict[str, Any]] = []
    robust_switches: list[dict[str, Any]] = []
    adjacency: list[dict[str, Any]] = []
    for low, high, axis_i in pairs:
        axis = axes[axis_i]
        adjacency.append(
            {
                "axis": axis["name"],
                "from_point_id": low["point_id"],
                "to_point_id": high["point_id"],
            }
        )
        central_switches.extend(
            membership_switches(low, high, axis, candidate_ids, "central")
        )
        robust_switches.extend(
            membership_switches(low, high, axis, candidate_ids, "robust")
        )

    candidate_regions: dict[str, dict[str, list[str]]] = {}
    for cid in sorted(candidate_ids):
        candidate_regions[cid] = {
            "central_front_points": sorted(
                p["point_id"] for p in normalized_points if cid in p["central_front"]
            ),
            "robust_front_points": sorted(
                p["point_id"] for p in normalized_points if cid in p["robust_front"]
            ),
        }

    output = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "pass",
        "sweep_name": sweep_name,
        "study_family_id": study_family,
        "source_manifest_sha256": sha256_file(manifest_path),
        "candidate_catalog": catalog,
        "candidate_catalog_sha256": catalog_hash,
        "objective_policy_sha256": reference_policy,
        "objectives": json.loads(reference_objectives.decode("utf-8")) if reference_objectives else [],
        "units": json.loads(reference_units.decode("utf-8")) if reference_units else {},
        "axes": axes,
        "sampled_point_count": len(normalized_points),
        "full_grid_point_count": math.prod(len(axis["values"]) for axis in axes),
        "sampled_points": sorted(normalized_points, key=lambda p: p["point_id"]),
        "adjacent_sampled_edges": sorted(
            adjacency,
            key=lambda e: (e["axis"], e["from_point_id"], e["to_point_id"]),
        ),
        "candidate_front_regions": candidate_regions,
        "central_switch_brackets": sorted(
            central_switches,
            key=lambda x: (x["axis"], x["from_point_id"], x["to_point_id"], x["candidate_id"]),
        ),
        "robust_switch_brackets": sorted(
            robust_switches,
            key=lambda x: (x["axis"], x["from_point_id"], x["to_point_id"], x["candidate_id"]),
        ),
        "interpolation_policy": "forbidden_v1",
        "switching_semantics": (
            "Switch brackets are observed only between adjacent declared grid values "
            "with both endpoint analyses present. Missing samples are never bridged, "
            "and no exact crossover threshold is inferred."
        ),
        "non_claims": [
            "Observed switch brackets do not identify an exact physical or economic crossover.",
            "Pareto-front membership changes do not select a universal transport architecture.",
            "No interpolation or probability of superiority is inferred in LL-009H v1.",
        ],
    }
    output["analysis_sha256"] = sha256_bytes(canonical_json_bytes(output))
    return output


def write_analysis(path: Path, central: list[str], robust: list[str], policy_hash: str) -> None:
    result = {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "pass",
        "study_id": "study-" + path.stem,
        "accounting_contract_id": "accounting-" + path.stem,
        "objective_policy_sha256": policy_hash,
        "objective_policy_name": "synthetic",
        "objectives": [{"metric": "m", "direction": "minimize", "domain": "nonnegative"}],
        "units": {"m": "u"},
        "candidate_vectors": {
            "A": {"m": {"low": 0.0, "central": 1.0, "high": 2.0, "unit": "u"}},
            "B": {"m": {"low": 0.0, "central": 1.0, "high": 2.0, "unit": "u"}},
        },
        "report_only_metrics": {"A": {}, "B": {}},
        "central_dominance_edges": [],
        "central_front": central,
        "robust_dominance_edges": [],
        "robust_front": robust,
        "central_not_robust_edges": [],
        "front_stability": {},
        "uncertainty_semantics": "synthetic",
        "non_claims": ["synthetic"],
    }
    result["analysis_sha256"] = sha256_bytes(canonical_json_bytes(result))
    path.write_bytes(canonical_json_bytes(result))


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ll009h-selftest-") as tmp:
        root = Path(tmp)
        analyses = root / "analyses"
        analyses.mkdir()
        policy_hash = "a" * 64
        cases = [
            ("p100", 100.0, ["A"], ["A"]),
            ("p1000", 1000.0, ["A", "B"], ["A", "B"]),
            ("p10000", 10000.0, ["B"], ["A", "B"]),
        ]
        points = []
        for pid, throughput, central, robust in cases:
            path = analyses / f"{pid}.json"
            write_analysis(path, central, robust, policy_hash)
            points.append(
                {
                    "point_id": pid,
                    "coordinates": {"throughput_kg_per_year": throughput},
                    "analysis_path": path.name,
                    "sha256": sha256_file(path),
                }
            )
        catalog = [
            {"candidate_id": "A", "architecture": "continuous_track"},
            {"candidate_id": "B", "architecture": "ballistic_freight"},
        ]
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "sweep_name": "synthetic-throughput",
            "study_family_id": "family-synthetic",
            "interpolation_policy": "forbidden_v1",
            "axes": [
                {
                    "name": "throughput_kg_per_year",
                    "kind": "numeric",
                    "unit": "kg/year",
                    "values": [100.0, 1000.0, 10000.0],
                }
            ],
            "candidate_catalog": catalog,
            "candidate_catalog_sha256": sha256_bytes(canonical_json_bytes(sorted(catalog, key=lambda x: x["candidate_id"]))),
            "points": points,
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        result = analyze_sweep(manifest_path, analyses)
        assert result["status"] == "pass"
        assert result["sampled_point_count"] == 3
        assert {
            "candidate_id": "B",
            "front": "central",
            "event": "enter",
            "axis": "throughput_kg_per_year",
            "unit": "kg/year",
            "from_value": 100.0,
            "to_value": 1000.0,
            "from_point_id": "p100",
            "to_point_id": "p1000",
            "fixed_coordinates": {},
            "semantics": "observed_adjacent_grid_bracket_no_interpolation",
        } in result["central_switch_brackets"]
        assert any(
            x["candidate_id"] == "A"
            and x["event"] == "leave"
            and x["from_value"] == 1000.0
            and x["to_value"] == 10000.0
            for x in result["central_switch_brackets"]
        )
        assert any(
            x["candidate_id"] == "B" and x["event"] == "enter"
            for x in result["robust_switch_brackets"]
        )
        assert not any(
            x["candidate_id"] == "A" and x["event"] == "leave"
            for x in result["robust_switch_brackets"]
        )
        serialized = canonical_json_bytes(result)
        assert b'"threshold"' not in serialized
        assert b'"winner"' not in serialized

        duplicate = json.loads(json.dumps(manifest))
        duplicate["points"].append(dict(duplicate["points"][0], point_id="duplicate"))
        duplicate_path = root / "duplicate.json"
        duplicate_path.write_bytes(canonical_json_bytes(duplicate))
        try:
            analyze_sweep(duplicate_path, analyses)
        except SweepError:
            pass
        else:
            raise AssertionError("duplicate coordinate must fail")

        mismatch_path = analyses / "p1000.json"
        mismatch = load_json(mismatch_path)
        mismatch["objective_policy_sha256"] = "b" * 64
        mismatch.pop("analysis_sha256", None)
        mismatch["analysis_sha256"] = sha256_bytes(canonical_json_bytes(mismatch))
        mismatch_path.write_bytes(canonical_json_bytes(mismatch))
        mismatch_manifest = json.loads(json.dumps(manifest))
        for point in mismatch_manifest["points"]:
            if point["point_id"] == "p1000":
                point["sha256"] = sha256_file(mismatch_path)
        mismatch_manifest_path = root / "mismatch.json"
        mismatch_manifest_path.write_bytes(canonical_json_bytes(mismatch_manifest))
        try:
            analyze_sweep(mismatch_manifest_path, analyses)
        except SweepError:
            pass
        else:
            raise AssertionError("objective-policy mismatch must fail")

        write_analysis(mismatch_path, ["A", "B"], ["A", "B"], policy_hash)
        sparse = json.loads(json.dumps(manifest))
        sparse["points"] = [sparse["points"][0], sparse["points"][2]]
        sparse_path = root / "sparse.json"
        sparse_path.write_bytes(canonical_json_bytes(sparse))
        sparse_result = analyze_sweep(sparse_path, analyses)
        assert sparse_result["adjacent_sampled_edges"] == []
        assert sparse_result["central_switch_brackets"] == []
        assert sparse_result["robust_switch_brackets"] == []


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--analysis-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        print("LL-009H scenario sweep self-test: PASS")
        return 0
    if args.manifest is None or args.analysis_root is None:
        raise SweepError(
            "--manifest and --analysis-root are required unless --self-test is used"
        )
    result = analyze_sweep(args.manifest, args.analysis_root)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SweepError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
