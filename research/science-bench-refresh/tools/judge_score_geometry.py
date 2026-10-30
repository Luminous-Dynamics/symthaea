#!/usr/bin/env python3
"""Matched judge/human score-space diagnostics (standard library only).

This computes descriptive fixed-set geometry; it does not score candidates,
establish human preference, prove construct validity, or estimate generalized
accuracy. Bootstrap resampling is clustered by item.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any

SCHEMA = "symthaea.science_bench_refresh_judge_score_geometry_input.v1"
REPORT_SCHEMA = "symthaea.science_bench_refresh_judge_score_geometry_report.v1"
DIGEST64 = re.compile(r"^[0-9a-f]{64}$")
REVISION40 = re.compile(r"^[0-9a-f]{40}$")
TOP_FIELDS = {
    "schema", "purpose", "analysis_id", "judge_source_revision",
    "human_protocol_sha256", "rubric_sha256", "criterion_ids",
    "analysis_config", "items",
}
CONFIG_FIELDS = {
    "bootstrap_resamples", "min_items_for_bootstrap",
    "min_items_per_task_family", "jacobi_tolerance",
}
ITEM_FIELDS = {"item_id", "task_family", "judge_ratings", "human_ratings"}
RATING_FIELDS = {"rater_id_sha256", "scores_millionths"}
MAX_ITEMS = 100_000
MAX_RATERS_PER_ITEM = 128
MAX_CRITERIA = 32
MAX_BOOTSTRAP = 10_000


class GeometryInputError(ValueError):
    pass


def _object(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GeometryInputError(f"{context} must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise GeometryInputError(f"{context} missing fields: {','.join(missing)}")
    if unknown:
        raise GeometryInputError(f"{context} unknown fields: {','.join(unknown)}")
    return value


def validate_input(data: Any) -> dict[str, Any]:
    value = _object(data, TOP_FIELDS, "input")
    if value["schema"] != SCHEMA:
        raise GeometryInputError("unsupported score geometry input schema")
    if value["purpose"] not in {"example_only", "qualified_measurement"}:
        raise GeometryInputError("purpose must be example_only or qualified_measurement")
    example = value["purpose"] == "example_only"
    for field in ("judge_source_revision",):
        revision = value[field]
        if not isinstance(revision, str) or not revision.strip():
            raise GeometryInputError(f"{field} must be non-empty")
        if not example and (REVISION40.fullmatch(revision) is None or set(revision) == {"0"}):
            raise GeometryInputError(f"qualified measurement requires nonzero {field}")
    for field in ("human_protocol_sha256", "rubric_sha256"):
        digest = value[field]
        if not isinstance(digest, str) or DIGEST64.fullmatch(digest) is None:
            raise GeometryInputError(f"{field} must be 64 lowercase hexadecimal characters")
        if not example and set(digest) == {"0"}:
            raise GeometryInputError(f"qualified measurement cannot use a zero {field}")
    analysis_id = value["analysis_id"]
    if not isinstance(analysis_id, str) or not analysis_id.strip() or len(analysis_id) > 128:
        raise GeometryInputError("analysis_id must be non-empty and at most 128 characters")

    criteria = value["criterion_ids"]
    if not isinstance(criteria, list) or not 2 <= len(criteria) <= MAX_CRITERIA:
        raise GeometryInputError(f"criterion_ids must contain 2..={MAX_CRITERIA} entries")
    if any(not isinstance(item, str) or not item.strip() for item in criteria):
        raise GeometryInputError("criterion_ids must be non-empty strings")
    if len(criteria) != len(set(criteria)):
        raise GeometryInputError("criterion_ids contains duplicates")

    config = _object(value["analysis_config"], CONFIG_FIELDS, "analysis_config")
    for name, minimum, maximum in (
        ("bootstrap_resamples", 0, MAX_BOOTSTRAP),
        ("min_items_for_bootstrap", 20, MAX_ITEMS),
        ("min_items_per_task_family", 2, MAX_ITEMS),
    ):
        number = config[name]
        if not isinstance(number, int) or isinstance(number, bool) or not minimum <= number <= maximum:
            raise GeometryInputError(f"{name} must be an integer in {minimum}..={maximum}")
    tolerance = config["jacobi_tolerance"]
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise GeometryInputError("jacobi_tolerance must be numeric")
    if not math.isfinite(tolerance) or not 1e-15 <= tolerance <= 1e-3:
        raise GeometryInputError("jacobi_tolerance must be finite and in 1e-15..=1e-3")

    items = value["items"]
    if not isinstance(items, list) or not 3 <= len(items) <= MAX_ITEMS:
        raise GeometryInputError(f"items must contain 3..={MAX_ITEMS} rows")
    item_ids: set[str] = set()
    seen_judges: set[str] = set()
    seen_humans: set[str] = set()
    for index, item in enumerate(items):
        row = _object(item, ITEM_FIELDS, f"items[{index}]")
        item_id = row["item_id"]
        family = row["task_family"]
        if not isinstance(item_id, str) or not item_id.strip() or len(item_id) > 128:
            raise GeometryInputError(f"items[{index}].item_id must be non-empty")
        if item_id in item_ids:
            raise GeometryInputError(f"duplicate item_id {item_id}")
        item_ids.add(item_id)
        if not isinstance(family, str) or not family.strip() or len(family) > 128:
            raise GeometryInputError(f"items[{index}].task_family must be non-empty")
        for panel_name, panel, global_ids in (
            ("judge_ratings", row["judge_ratings"], seen_judges),
            ("human_ratings", row["human_ratings"], seen_humans),
        ):
            if not isinstance(panel, list) or not 2 <= len(panel) <= MAX_RATERS_PER_ITEM:
                raise GeometryInputError(f"items[{index}].{panel_name} must contain 2..={MAX_RATERS_PER_ITEM} raters")
            local_ids: set[str] = set()
            for rating_index, rating in enumerate(panel):
                rating = _object(rating, RATING_FIELDS, f"items[{index}].{panel_name}[{rating_index}]")
                rater_id = rating["rater_id_sha256"]
                if not isinstance(rater_id, str) or DIGEST64.fullmatch(rater_id) is None or set(rater_id) == {"0"}:
                    raise GeometryInputError(f"items[{index}].{panel_name}[{rating_index}] has invalid rater id digest")
                if rater_id in local_ids:
                    raise GeometryInputError(f"items[{index}].{panel_name} contains duplicate rater id")
                local_ids.add(rater_id)
                global_ids.add(rater_id)
                scores = rating["scores_millionths"]
                if not isinstance(scores, list) or len(scores) != len(criteria):
                    raise GeometryInputError(f"rater {rater_id} score vector must match criterion_ids length")
                for score in scores:
                    if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 1_000_000:
                        raise GeometryInputError(f"rater {rater_id} scores must be integer millionths in 0..1,000,000")
    if seen_judges & seen_humans:
        raise GeometryInputError("judge and human pseudonymous rater IDs must be disjoint")
    return value


def _flatten_panel(items: list[dict[str, Any]], panel_name: str) -> list[float]:
    return [
        float(score)
        for item in items
        for rating in item[panel_name]
        for score in rating["scores_millionths"]
    ]


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.pstdev(values)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    xx = sum(value * value for value in dx)
    yy = sum(value * value for value in dy)
    if xx <= 0.0 or yy <= 0.0:
        return None
    result = sum(x * y for x, y in zip(dx, dy)) / math.sqrt(xx * yy)
    return max(-1.0, min(1.0, result))


def _rater_maps(items: list[dict[str, Any]], panel_name: str) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    for item in items:
        for rating in item[panel_name]:
            result.setdefault(rating["rater_id_sha256"], {})[item["item_id"]] = [
                float(value) for value in rating["scores_millionths"]
            ]
    return result


def _pairwise_summary(
    items: list[dict[str, Any]], left_name: str, right_name: str
) -> dict[str, Any]:
    left = _rater_maps(items, left_name)
    right = _rater_maps(items, right_name)
    same_panel = left_name == right_name
    correlations: list[float] = []
    pair_count = 0
    undefined = 0
    left_ids = sorted(left)
    right_ids = sorted(right)
    for i, left_id in enumerate(left_ids):
        start = i + 1 if same_panel else 0
        for right_id in right_ids[start:]:
            pair_count += 1
            shared_items = sorted(set(left[left_id]) & set(right[right_id]))
            xs: list[float] = []
            ys: list[float] = []
            for item_id in shared_items:
                xs.extend(left[left_id][item_id])
                ys.extend(right[right_id][item_id])
            corr = _correlation(xs, ys)
            if corr is None:
                undefined += 1
            else:
                correlations.append(corr)
    return {
        "pair_count": pair_count,
        "defined_pair_count": len(correlations),
        "undefined_pair_count": undefined,
        "mean_correlation": statistics.fmean(correlations) if correlations else None,
        "median_correlation": statistics.median(correlations) if correlations else None,
        "minimum_correlation": min(correlations) if correlations else None,
        "maximum_correlation": max(correlations) if correlations else None,
        "correlations": correlations,
    }


def _panel_mean_matrix(items: list[dict[str, Any]], panel_name: str) -> list[list[float]]:
    matrix: list[list[float]] = []
    for item in items:
        ratings = item[panel_name]
        criterion_count = len(ratings[0]["scores_millionths"])
        row = []
        for index in range(criterion_count):
            row.append(statistics.fmean(float(rating["scores_millionths"][index]) for rating in ratings))
        matrix.append(row)
    return matrix


def _covariance(matrix: list[list[float]]) -> list[list[float]]:
    row_count = len(matrix)
    dimensions = len(matrix[0]) if matrix else 0
    means = [statistics.fmean(row[column] for row in matrix) for column in range(dimensions)]
    centered = [[row[column] - means[column] for column in range(dimensions)] for row in matrix]
    divisor = max(1, row_count - 1)
    return [
        [
            sum(row[i] * row[j] for row in centered) / divisor
            for j in range(dimensions)
        ]
        for i in range(dimensions)
    ]


def _jacobi_eigen(
    matrix: list[list[float]], tolerance: float
) -> tuple[list[float], list[list[float]]]:
    """Eigenvalues and eigenvectors (columns) of a real symmetric matrix."""
    n = len(matrix)
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    if n == 0:
        return [], []
    max_iterations = max(1, 100 * n * n)
    converged = False
    for _ in range(max_iterations):
        p, q = 0, 0
        largest = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                value = abs(a[i][j])
                if value > largest:
                    largest, p, q = value, i, j
        if largest <= tolerance:
            converged = True
            break
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        angle = 0.5 * math.atan2(2.0 * apq, aqq - app)
        c, s = math.cos(angle), math.sin(angle)
        for k in range(n):
            if k in (p, q):
                continue
            akp, akq = a[k][p], a[k][q]
            a[k][p] = a[p][k] = c * akp - s * akq
            a[k][q] = a[q][k] = s * akp + c * akq
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        a[p][q] = a[q][p] = 0.0
        for k in range(n):
            vkp, vkq = v[k][p], v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq
    if not converged:
        raise GeometryInputError("Jacobi eigensolver did not converge within its deterministic iteration bound")
    eigenvalues = [max(0.0, a[i][i]) for i in range(n)]
    order = sorted(range(n), key=lambda index: eigenvalues[index], reverse=True)
    sorted_values = [eigenvalues[index] for index in order]
    sorted_vectors = [[v[row][index] for index in order] for row in range(n)]
    return sorted_values, sorted_vectors


def _effective_rank(eigenvalues: list[float]) -> float | None:
    total = sum(eigenvalues)
    if total <= 0.0:
        return None
    probabilities = [value / total for value in eigenvalues if value > 0.0]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    return math.exp(entropy)


def _matrix_geometry(
    judge_matrix: list[list[float]],
    human_matrix: list[list[float]],
    tolerance: float,
) -> dict[str, Any]:
    if len(judge_matrix) < 2 or len(human_matrix) < 2:
        return {
            "judge_effective_rank": None, "human_effective_rank": None,
            "leading_axis_angle_degrees": None, "eigensolver_converged": False,
        }
    judge_values, judge_vectors = _jacobi_eigen(_covariance(judge_matrix), tolerance)
    human_values, human_vectors = _jacobi_eigen(_covariance(human_matrix), tolerance)
    judge_axis = [judge_vectors[row][0] for row in range(len(judge_vectors))]
    human_axis = [human_vectors[row][0] for row in range(len(human_vectors))]
    dot = abs(sum(left * right for left, right in zip(judge_axis, human_axis)))
    angle = math.degrees(math.acos(max(0.0, min(1.0, dot))))
    return {
        "judge_effective_rank": _effective_rank(judge_values),
        "human_effective_rank": _effective_rank(human_values),
        "leading_axis_angle_degrees": angle,
        "eigensolver_converged": True,
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap(
    judge_matrix: list[list[float]],
    human_matrix: list[list[float]],
    input_seed: int,
    resamples: int,
    tolerance: float,
) -> dict[str, Any]:
    count = len(judge_matrix)
    if count < 2 or resamples <= 0:
        return {
            "planned_resamples": resamples, "valid_resamples": 0,
            "score_spread_ratio_ci_95": None, "leading_axis_angle_ci_95": None,
            "not_evaluable_reason": "bootstrap disabled or insufficient item count",
        }
    rng = random.Random(input_seed)
    spread: list[float] = []
    angles: list[float] = []
    for _ in range(resamples):
        indices = [rng.randrange(count) for _ in range(count)]
        judges = [judge_matrix[index] for index in indices]
        humans = [human_matrix[index] for index in indices]
        ratio = _ratio(
            _std([value for row in judges for value in row]),
            _std([value for row in humans for value in row]),
        )
        if ratio is not None:
            spread.append(ratio)
        try:
            geometry = _matrix_geometry(judges, humans, tolerance)
            angle = geometry["leading_axis_angle_degrees"]
            if angle is not None and math.isfinite(angle):
                angles.append(angle)
        except GeometryInputError:
            continue
    return {
        "planned_resamples": resamples,
        "valid_spread_resamples": len(spread),
        "valid_angle_resamples": len(angles),
        "score_spread_ratio_ci_95": [_percentile(spread, 0.025), _percentile(spread, 0.975)] if spread else None,
        "leading_axis_angle_ci_95": [_percentile(angles, 0.025), _percentile(angles, 0.975)] if angles else None,
        "resampling_unit": "item_cluster",
        "not_evaluable_reason": None if spread and angles else "one or more metrics had no valid bootstrap resamples",
    }


def analyze_score_geometry(input_bytes: bytes) -> dict[str, Any]:
    try:
        raw = json.loads(input_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise GeometryInputError(f"input JSON malformed: {error}") from error
    data = validate_input(raw)
    items = data["items"]
    judge_matrix = _panel_mean_matrix(items, "judge_ratings")
    human_matrix = _panel_mean_matrix(items, "human_ratings")
    judge_pooled = _flatten_panel(items, "judge_ratings")
    human_pooled = _flatten_panel(items, "human_ratings")
    judge_std = _std(judge_pooled)
    human_std = _std(human_pooled)
    judge_panel_std = _std([value for row in judge_matrix for value in row])
    human_panel_std = _std([value for row in human_matrix for value in row])
    config = data["analysis_config"]
    input_digest = hashlib.sha256(input_bytes).hexdigest()
    seed = int.from_bytes(hashlib.sha256(input_bytes + b"\0judge-score-geometry-bootstrap-v1").digest()[:8], "big")

    pairwise = {
        "judge_judge": _pairwise_summary(items, "judge_ratings", "judge_ratings"),
        "judge_human": _pairwise_summary(items, "judge_ratings", "human_ratings"),
        "human_human": _pairwise_summary(items, "human_ratings", "human_ratings"),
    }
    geometry = _matrix_geometry(judge_matrix, human_matrix, config["jacobi_tolerance"])
    bootstrap = None
    if len(items) >= config["min_items_for_bootstrap"]:
        bootstrap = _bootstrap(
            judge_matrix, human_matrix, seed,
            config["bootstrap_resamples"], config["jacobi_tolerance"],
        )
    else:
        bootstrap = {
            "planned_resamples": config["bootstrap_resamples"],
            "valid_spread_resamples": 0,
            "valid_angle_resamples": 0,
            "score_spread_ratio_ci_95": None,
            "leading_axis_angle_ci_95": None,
            "resampling_unit": "item_cluster",
            "not_evaluable_reason": "item count below min_items_for_bootstrap",
        }

    families: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        families.setdefault(item["task_family"], []).append(item)
    family_summaries = []
    for family, rows in sorted(families.items()):
        if len(rows) < config["min_items_per_task_family"]:
            family_summaries.append({
                "task_family": family, "item_count": len(rows),
                "evaluable": False, "not_evaluable_reason": "below min_items_per_task_family",
                "score_spread_ratio": None, "leading_axis_angle_degrees": None,
            })
            continue
        family_judges = _panel_mean_matrix(rows, "judge_ratings")
        family_humans = _panel_mean_matrix(rows, "human_ratings")
        family_s = _ratio(
            _std([value for row in family_judges for value in row]),
            _std([value for row in family_humans for value in row]),
        )
        family_geometry = _matrix_geometry(family_judges, family_humans, config["jacobi_tolerance"])
        family_summaries.append({
            "task_family": family, "item_count": len(rows), "evaluable": True,
            "not_evaluable_reason": None,
            "score_spread_ratio": family_s,
            "leading_axis_angle_degrees": family_geometry["leading_axis_angle_degrees"],
            "judge_effective_rank": family_geometry["judge_effective_rank"],
            "human_effective_rank": family_geometry["human_effective_rank"],
        })

    return {
        "schema": REPORT_SCHEMA,
        "analysis_id": data["analysis_id"],
        "input_sha256": input_digest,
        "valid": True,
        "item_count": len(items),
        "criterion_count": len(data["criterion_ids"]),
        "criterion_ids": data["criterion_ids"],
        "pairwise_agreement": pairwise,
        "pooled_rater_score_spread_ratio": _ratio(judge_std, human_std),
        "judge_panel_mean_score_spread_ratio": _ratio(judge_panel_std, human_panel_std),
        "judge_pooled_score_std_millionths": judge_std,
        "human_pooled_score_std_millionths": human_std,
        "judge_panel_mean_score_std_millionths": judge_panel_std,
        "human_panel_mean_score_std_millionths": human_panel_std,
        **geometry,
        "bootstrap": bootstrap,
        "task_family_summaries": family_summaries,
        "fixed_score_matrix_only": True,
        "generalized_accuracy_estimated": False,
        "item_population_sampling_claimed": False,
        "irt_model_fitted": False,
        "construct_validity_proven": False,
        "judge_quality_established": False,
        "external_anchor_required": True,
        "scoring_eligible_declared": False,
        "claim_boundary": "Descriptive geometry of the supplied matched fixed score matrix only; no generalized accuracy, human preference validity, causal effect, IRT estimate, external custody or scoring qualification is established.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        payload = args.input.read_bytes()
        report = analyze_score_geometry(payload)
    except (OSError, GeometryInputError) as error:
        output = json.dumps({"valid": False, "issues": [str(error)]}, indent=2) + "\n"
        print(output, end="")
        return 2
    output = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
