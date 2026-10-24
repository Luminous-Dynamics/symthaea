#!/usr/bin/env python3
"""Independently recompute v1 two-objective hypervolume metrics from hashed JSONL.

Benchmark-only standard-library tool. This deliberately supports only single-fidelity,
two-objective minimization traces; unsupported modes fail closed rather than being guessed.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    from validate_receipt import validate_receipt
except ImportError:  # pragma: no cover - supports package-style execution too
    from .validate_receipt import validate_receipt

TRACE_SCHEMA_VERSION = "1.0.0"
ABS_TOLERANCE = 1e-9
TRACE_FIELDS = {"schema_version", "evaluation_index", "candidate_id", "status", "objectives"}
TRACE_STATUSES = {"feasible", "infeasible", "invalid", "failed"}


class TraceScoreError(ValueError):
    """A receipt/trace mismatch or an unsupported benchmark convention."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_constant(value: str) -> None:
    raise TraceScoreError(f"non-standard JSON numeric constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TraceScoreError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _strict_json_loads(text: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except TraceScoreError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise TraceScoreError(f"invalid JSON: {exc}") from exc


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def hypervolume_2d_minimize(points: list[list[float]], reference: list[float]) -> float:
    """Exact dominated area for 2D minimization, discarding points outside the reference box."""
    if len(reference) != 2 or not all(_finite_number(v) for v in reference):
        raise TraceScoreError("reference point must contain two finite coordinates")
    rx, ry = float(reference[0]), float(reference[1])
    eligible = [
        (float(p[0]), float(p[1]))
        for p in points
        if len(p) == 2
        and all(_finite_number(v) for v in p)
        and float(p[0]) < rx
        and float(p[1]) < ry
    ]
    # Sort by x ascending; for equal x, the lowest y comes first. Keep only strict
    # improvements in y, leaving a non-dominated front with x ascending and y descending.
    eligible.sort(key=lambda point: (point[0], point[1]))
    front: list[tuple[float, float]] = []
    best_y = math.inf
    for x, y in eligible:
        if y < best_y:
            front.append((x, y))
            best_y = y

    area = 0.0
    for index, (x, y) in enumerate(front):
        next_x = front[index + 1][0] if index + 1 < len(front) else rx
        width = next_x - x
        height = ry - y
        if width < 0.0 or height < 0.0:
            raise TraceScoreError("internal hypervolume ordering invariant failed")
        area += width * height
    if not math.isfinite(area):
        raise TraceScoreError("hypervolume overflowed to a non-finite result")
    return area


class HypervolumeArchive2D:
    """Incremental exact 2D minimization archive; each inserted point is handled amortized O(n)."""

    def __init__(self, reference: list[float]) -> None:
        if len(reference) != 2 or not all(_finite_number(v) for v in reference):
            raise TraceScoreError("reference point must contain two finite coordinates")
        self.reference = [float(reference[0]), float(reference[1])]
        self._front: list[tuple[float, float]] = []
        self._xs: list[float] = []
        self._area = 0.0

    @property
    def front(self) -> list[list[float]]:
        return [[x, y] for x, y in self._front]

    @property
    def area(self) -> float:
        return self._area

    def observe(self, point: list[float]) -> float:
        if len(point) != 2 or not all(_finite_number(v) for v in point):
            raise TraceScoreError("observed point must contain two finite coordinates")
        x, y = float(point[0]), float(point[1])
        rx, ry = self.reference
        if x >= rx or y >= ry:
            return self._area

        position = bisect.bisect_left(self._xs, x)
        # A same-x point with a better/equal y, or a predecessor with better/equal y,
        # dominates this candidate. The x-ordered front has strictly decreasing y values.
        if position < len(self._front):
            same_x, same_y = self._front[position]
            if same_x == x and same_y <= y:
                return self._area
        if position > 0 and self._front[position - 1][1] <= y:
            return self._area

        predecessor_delta = 0.0
        if position > 0:
            predecessor_x, predecessor_y = self._front[position - 1]
            old_next_x = self._xs[position] if position < len(self._xs) else rx
            predecessor_delta = (x - old_next_x) * (ry - predecessor_y)

        # The new point dominates the consecutive following points whose y is no better.
        end = position
        while end < len(self._front) and self._front[end][1] >= y:
            end += 1

        removed_area_parts = []
        for index in range(position, end):
            old_x, old_y = self._front[index]
            old_next_x = self._xs[index + 1] if index + 1 < len(self._xs) else rx
            removed_area_parts.append((old_next_x - old_x) * (ry - old_y))
        removed_area = math.fsum(removed_area_parts)
        next_survivor_x = self._xs[end] if end < len(self._xs) else rx
        new_area_part = (next_survivor_x - x) * (ry - y)
        updated_area = math.fsum((self._area, predecessor_delta, new_area_part, -removed_area))
        if not math.isfinite(updated_area) or updated_area < -ABS_TOLERANCE:
            raise TraceScoreError("incremental hypervolume invariant failed")
        self._area = max(0.0, updated_area)

        del self._front[position:end]
        del self._xs[position:end]
        self._front.insert(position, (x, y))
        self._xs.insert(position, x)
        return self._area


def _parse_trace(trace_bytes: bytes) -> list[dict[str, Any]]:
    try:
        text = trace_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise TraceScoreError(f"trace is not valid UTF-8: {exc}") from exc
    if not text:
        raise TraceScoreError("trace must contain at least one evaluation row")
    if not text.endswith("\n"):
        raise TraceScoreError("trace must end with one LF newline for canonical hashing")
    if "\r" in text:
        raise TraceScoreError("trace must use canonical LF newlines, not CR/CRLF")
    lines = text.split("\n")[:-1]
    if not lines or any(not line.strip() for line in lines):
        raise TraceScoreError("blank JSONL rows are forbidden")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        value = _strict_json_loads(line)
        if not isinstance(value, dict):
            raise TraceScoreError(f"line {line_number}: expected a JSON object")
        if set(value) != TRACE_FIELDS:
            missing = sorted(TRACE_FIELDS - value.keys())
            extra = sorted(value.keys() - TRACE_FIELDS)
            raise TraceScoreError(
                f"line {line_number}: trace fields mismatch; missing={missing}, extra={extra}"
            )
        if value["schema_version"] != TRACE_SCHEMA_VERSION:
            raise TraceScoreError(f"line {line_number}: unsupported trace schema version")
        if isinstance(value["evaluation_index"], bool) or not isinstance(value["evaluation_index"], int):
            raise TraceScoreError(f"line {line_number}: evaluation_index must be an integer")
        if value["evaluation_index"] != line_number:
            raise TraceScoreError(
                f"line {line_number}: evaluation_index must be contiguous and 1-based"
            )
        if not isinstance(value["candidate_id"], str) or not value["candidate_id"].strip():
            raise TraceScoreError(f"line {line_number}: candidate_id must be a non-empty string")
        if not isinstance(value["status"], str) or value["status"] not in TRACE_STATUSES:
            raise TraceScoreError(f"line {line_number}: invalid evaluation status")

        objectives = value["objectives"]
        if value["status"] == "feasible":
            if not isinstance(objectives, list) or len(objectives) != 2:
                raise TraceScoreError(f"line {line_number}: feasible evaluations require two objectives")
        elif value["status"] in {"invalid", "failed"} and objectives is not None:
            raise TraceScoreError(f"line {line_number}: invalid/failed evaluations require objectives=null")
        if objectives is not None:
            if not isinstance(objectives, list) or len(objectives) != 2:
                raise TraceScoreError(f"line {line_number}: objectives must be null or a two-number array")
            if not all(_finite_number(v) for v in objectives):
                raise TraceScoreError(f"line {line_number}: objective values must be finite numbers")
        rows.append(value)
    return rows


def score_trace(
    receipt: dict[str, Any],
    trace_bytes: bytes,
    trace_schema_bytes: bytes,
    scorer_bytes: bytes,
    protocol_bytes: bytes,
    analysis_lock_bytes: bytes,
    for_publication: bool = False,
) -> dict[str, Any]:
    """Validate receipt identity/accounting, verify digests and independently recompute metrics."""
    receipt_errors = validate_receipt(receipt, for_publication=for_publication)
    if receipt_errors:
        raise TraceScoreError("invalid receipt: " + "; ".join(receipt_errors))

    # Bind the run to the exact protocol and analysis-lock bytes in this checkout.
    if receipt["protocol_sha256"] != sha256_hex(protocol_bytes):
        raise TraceScoreError("protocol digest mismatch: protocol_sha256")
    if receipt["analysis_lock_sha256"] != sha256_hex(analysis_lock_bytes):
        raise TraceScoreError("analysis-lock digest mismatch: analysis_lock_sha256")
    protocol = _strict_json_loads(protocol_bytes.decode("utf-8", errors="strict"))
    analysis_lock = _strict_json_loads(analysis_lock_bytes.decode("utf-8", errors="strict"))
    if not isinstance(protocol, dict) or protocol.get("protocol_id") != receipt["protocol_id"]:
        raise TraceScoreError("protocol lock does not identify the receipt protocol")
    raw_trace_contract = protocol.get("raw_trace_contract")
    if not isinstance(raw_trace_contract, dict):
        raise TraceScoreError("protocol lock is missing its raw-trace contract")
    if raw_trace_contract.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise TraceScoreError("protocol lock does not declare the frozen trace schema version")
    if raw_trace_contract.get("analysis_lock_file") != "docs/research/CASTING_BENCHMARK_ANALYSIS_LOCK_V1.json":
        raise TraceScoreError("protocol lock does not point to the frozen analysis lock")
    if not isinstance(analysis_lock, dict) or analysis_lock.get("protocol_id") != receipt["protocol_id"]:
        raise TraceScoreError("analysis lock does not identify the receipt protocol")
    if analysis_lock.get("analysis_lock_version") != "1.0.0":
        raise TraceScoreError("unsupported analysis-lock version")
    lock_scorer = analysis_lock.get("scorer")
    if not isinstance(lock_scorer, dict):
        raise TraceScoreError("analysis lock is missing its scorer contract")
    if lock_scorer.get("path") != "tools/casting-benchmark/score_trace.py":
        raise TraceScoreError("scorer path differs from immutable analysis lock")
    if lock_scorer.get("trace_schema_path") != raw_trace_contract.get("trace_schema_file"):
        raise TraceScoreError("trace schema path differs between protocol and analysis lock")
    if lock_scorer.get("supported_evaluation_modes") != ["single_fidelity"]:
        raise TraceScoreError("unsupported evaluation modes in analysis lock")
    if lock_scorer.get("supported_objective_count") != 2 or lock_scorer.get("supported_objective_direction") != "minimize_all":
        raise TraceScoreError("unsupported objective contract in analysis lock")

    if receipt["evaluation_mode"] != "single_fidelity":
        raise TraceScoreError("v1 trace scorer only supports single_fidelity receipts")
    suite = receipt["suite"]
    suite_contracts = analysis_lock.get("suite_contracts")
    if not isinstance(suite_contracts, dict) or suite["id"] not in suite_contracts:
        raise TraceScoreError(f"suite is not present in immutable analysis lock: {suite['id']}")
    contract = suite_contracts[suite["id"]]
    if not isinstance(contract, dict):
        raise TraceScoreError("malformed suite contract in analysis lock")
    if contract.get("fixture_only") is True and receipt["synthetic_fixture_only"] is not True:
        raise TraceScoreError("test-only suite contract requires synthetic_fixture_only=true")
    if suite.get("objective_count") != contract.get("objective_count"):
        raise TraceScoreError("objective count differs from immutable suite contract")
    if suite.get("objective_direction") != contract.get("objective_direction"):
        raise TraceScoreError("objective direction differs from immutable suite contract")
    instance_contracts = contract.get("instances")
    instance_contract = instance_contracts.get(suite.get("instance_id")) if isinstance(instance_contracts, dict) else None
    if not isinstance(instance_contract, dict):
        raise TraceScoreError("suite instance is not present in immutable analysis lock")
    if suite.get("dimension") != instance_contract.get("decision_dimension"):
        raise TraceScoreError("decision dimension differs from immutable suite instance contract")
    if suite.get("ideal_point") != contract.get("ideal_point"):
        raise TraceScoreError("ideal point differs from immutable suite contract")
    if suite.get("reference_point") != contract.get("reference_point"):
        raise TraceScoreError("reference point differs from immutable suite contract")
    if suite["objective_count"] != 2 or suite["objective_direction"] != "minimize_all":
        raise TraceScoreError("v1 trace scorer only supports exactly two minimization objectives")

    ideal = suite["ideal_point"]
    reference = suite["reference_point"]
    if len(ideal) != 2 or len(reference) != 2:
        raise TraceScoreError("v1 trace scorer requires two-coordinate ideal and reference points")
    if not all(_finite_number(v) for v in ideal + reference):
        raise TraceScoreError("ideal/reference point must contain finite numbers")
    ideal = [float(v) for v in ideal]
    reference = [float(v) for v in reference]
    if any(lo >= hi for lo, hi in zip(ideal, reference)):
        raise TraceScoreError("each ideal coordinate must be below its reference coordinate")

    try:
        trace_schema = _strict_json_loads(trace_schema_bytes.decode("utf-8", errors="strict"))
    except UnicodeDecodeError as exc:
        raise TraceScoreError(f"trace schema is not valid UTF-8: {exc}") from exc
    schema_properties = trace_schema.get("properties") if isinstance(trace_schema, dict) else None
    schema_version_contract = schema_properties.get("schema_version") if isinstance(schema_properties, dict) else None
    if not isinstance(schema_version_contract, dict) or schema_version_contract.get("const") != TRACE_SCHEMA_VERSION:
        raise TraceScoreError("trace schema is missing the frozen v1.0.0 version contract")

    expected_normalization_area = (
        (reference[0] - ideal[0]) * (reference[1] - ideal[1])
    )
    locked_area = contract.get("normalization_area")
    if not _finite_number(locked_area) or not math.isclose(
        expected_normalization_area, float(locked_area), rel_tol=0.0, abs_tol=ABS_TOLERANCE
    ):
        raise TraceScoreError("normalization area differs from immutable suite contract")

    artifacts = receipt["artifacts"]
    expected_hashes = {
        "raw_trace_sha256": sha256_hex(trace_bytes),
        "trace_schema_sha256": sha256_hex(trace_schema_bytes),
        "scorer_sha256": sha256_hex(scorer_bytes),
    }
    for key, actual in expected_hashes.items():
        if artifacts.get(key) != actual:
            raise TraceScoreError(f"artifact digest mismatch: {key}")

    rows = _parse_trace(trace_bytes)
    accounting = receipt["accounting"]
    actual_count = len(rows)
    if actual_count != accounting["objective_evaluations"]:
        raise TraceScoreError("trace row count does not match accounting.objective_evaluations")
    if actual_count > receipt["budget"]["objective_evaluations"]:
        raise TraceScoreError("trace consumes more than the frozen objective-evaluation budget")
    if receipt["termination_reason"] == "budget_exhausted" and actual_count != receipt["budget"]["objective_evaluations"]:
        raise TraceScoreError("budget_exhausted requires actual evaluation count to equal planned budget")

    counts = {status: 0 for status in TRACE_STATUSES}
    archive = HypervolumeArchive2D(reference)
    hv_by_evaluation: list[float] = []
    for row in rows:
        counts[row["status"]] += 1
        objectives = row["objectives"]
        if objectives is not None:
            if any(float(value) < lower for value, lower in zip(objectives, ideal)):
                raise TraceScoreError(
                    f"evaluation {row['evaluation_index']}: objective value is below frozen ideal bound"
                )
        if row["status"] == "feasible":
            # The trace parser requires the objective vector for every feasible row.
            archive.observe([float(objectives[0]), float(objectives[1])])
        hv_by_evaluation.append(archive.area)

    for trace_status, receipt_key in (
        ("feasible", "feasible_evaluations"),
        ("infeasible", "infeasible_evaluations"),
        ("invalid", "invalid_evaluations"),
        ("failed", "failed_evaluations"),
    ):
        if counts[trace_status] != accounting[receipt_key]:
            raise TraceScoreError(
                f"trace status count {trace_status} does not match accounting.{receipt_key}"
            )

    actual_feasible_fraction = counts["feasible"] / actual_count
    actual_invalid_failed_fraction = (counts["invalid"] + counts["failed"]) / actual_count
    final_hypervolume = hv_by_evaluation[-1]

    computed_auc: float | None
    if receipt["status"] == "success":
        planned_budget = receipt["budget"]["objective_evaluations"]
        # Carry the last observed quality forward after a permitted early stop, so all
        # algorithms share the planned budget denominator without pretending calls occurred.
        area_sum = math.fsum(hv_by_evaluation)
        area_sum += final_hypervolume * (planned_budget - actual_count)
        normalization_area = (reference[0] - ideal[0]) * (reference[1] - ideal[1])
        if not math.isfinite(normalization_area) or normalization_area <= 0.0:
            raise TraceScoreError("invalid ideal/reference normalization area")
        computed_auc = area_sum / (planned_budget * normalization_area)
        if not math.isfinite(computed_auc) or computed_auc < -ABS_TOLERANCE or computed_auc > 1.0 + ABS_TOLERANCE:
            raise TraceScoreError("normalized hypervolume AUC is outside [0,1]; inspect fixed bounds/reference point")
    else:
        computed_auc = None

    recomputed = {
        "final_hypervolume": final_hypervolume,
        "normalized_hypervolume_auc": computed_auc,
        "feasible_fraction": actual_feasible_fraction,
        "invalid_or_failed_fraction": actual_invalid_failed_fraction,
    }
    if receipt["status"] == "success":
        declared = receipt["metrics"]
        for key, actual in recomputed.items():
            if actual is None:
                raise TraceScoreError(f"successful receipt has no independently computed {key}")
            reported = declared[key]
            if not _finite_number(reported) or not math.isclose(
                float(reported), actual, rel_tol=0.0, abs_tol=ABS_TOLERANCE
            ):
                raise TraceScoreError(
                    f"reported metric mismatch: {key} reports {reported!r}, trace recomputes {actual:.12g}"
                )

    return {
        "protocol_id": receipt["protocol_id"],
        "run_id": receipt["run_id"],
        "status": receipt["status"],
        "audited": True,
        "publishable_as_empirical": receipt["status"] == "success" and not receipt["synthetic_fixture_only"],
        "objective_evaluations": actual_count,
        "planned_budget": receipt["budget"]["objective_evaluations"],
        "status_counts": counts,
        "trace_sha256": expected_hashes["raw_trace_sha256"],
        "protocol_sha256": sha256_hex(protocol_bytes),
        "analysis_lock_sha256": sha256_hex(analysis_lock_bytes),
        "algorithm_id": receipt["algorithm"]["id"],
        "algorithm_version": receipt["algorithm"]["version"],
        "algorithm_configuration_sha256": receipt["algorithm"]["configuration_sha256"],
        "suite_id": suite["id"],
        "suite_version": suite["version"],
        "suite_instance_id": suite["instance_id"],
        "objective_evaluator_sha256": suite["objective_evaluator_sha256"],
        "source_revision": receipt["source_revision"],
        "environment_lock_sha256": receipt["environment_lock_sha256"],
        "analysis_lock_sha256": sha256_hex(analysis_lock_bytes),
        "planned_budget": receipt["budget"]["objective_evaluations"],
        "seed": receipt["seed"],
        "metrics_recomputed": recomputed,
        "reported_metrics_match": receipt["status"] == "success",
        "note": (
            "A successful receipt's metrics were independently recomputed and matched."
            if receipt["status"] == "success"
            else "Run is audited but is not an algorithm pass; no successful-result metrics were compared."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--trace-schema",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "research" / "CASTING_BENCHMARK_TRACE.schema.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "research" / "CASTING_BENCHMARK_PROTOCOL_V1.json",
    )
    parser.add_argument(
        "--analysis-lock",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "research" / "CASTING_BENCHMARK_ANALYSIS_LOCK_V1.json",
    )
    parser.add_argument("--for-publication", action="store_true")
    args = parser.parse_args()

    try:
        receipt = _strict_json_loads(args.receipt.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            raise TraceScoreError("receipt root must be an object")
        trace_bytes = args.trace.read_bytes()
        trace_schema_bytes = args.trace_schema.read_bytes()
        scorer_bytes = Path(__file__).read_bytes()
        protocol_bytes = args.protocol.read_bytes()
        analysis_lock_bytes = args.analysis_lock.read_bytes()
        result = score_trace(
            receipt,
            trace_bytes,
            trace_schema_bytes,
            scorer_bytes,
            protocol_bytes,
            analysis_lock_bytes,
            for_publication=args.for_publication,
        )
    except (OSError, UnicodeError, TraceScoreError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    if args.for_publication and not result["publishable_as_empirical"]:
        print("REJECTED: synthetic fixture cannot be published as empirical evidence", file=sys.stderr)
        return 1
    if result["status"] != "success":
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
