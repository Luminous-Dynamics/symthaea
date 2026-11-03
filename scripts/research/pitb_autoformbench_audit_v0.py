#!/usr/bin/env python3
"""Recompute the public AutoformBench residual audit from its pinned Git commit.

Reads only MANIFEST.csv and public task-level leaderboard CSVs. It does not
open textbook sources, Lean workspaces, accepted submissions, proofs, or traces.
This research audit is not a training-data importer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

PIN = "af1aa38fd44e019e566bce0b5a168609a886d8e6"
FILES = {
    "MANIFEST.csv": "45d5cac87088c19427b742072c6dc3deb2d6e433",
    "leaderboard_data/gpt-6-astra.csv": "d2994a38a2213c3ad1c240080e8dc531b77e64c8",
    "leaderboard_data/opus-5.csv": "f928544f58138bf1c0616c0fbfb1c7296f6bd923",
    "leaderboard_data/gpt-5.6-sol.csv": "b6f314bee4e20cfb1fdef8c6ca19f03dbc42249f",
    "leaderboard_data/deepseek-v4.1-flash.csv": "c9f226f7fdc0a0b536779cd1c6d7221ee39d3371",
}
SOLVE_COUNTS = {"gpt-6-astra": 281, "opus-5": 211, "gpt-5.6-sol": 201, "deepseek-v4.1-flash": 180}
NUMERIC = ("input_tokens", "cached_read_tokens", "cache_write_tokens", "output_tokens",
           "cost_usd", "turns", "compactions", "added_code_loc")


class AuditError(RuntimeError):
    pass


def read_git_file(repo: Path, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "show", f"{PIN}:{path}"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise AuditError(f"cannot read pinned object {path}; fetch the release commit first") from exc


def git_blob_sha1(raw: bytes) -> str:
    """Return Git's canonical SHA-1 for a blob, including its object header."""
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def parse_csv(data: bytes, label: str) -> list[dict[str, str]]:
    try:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise AuditError(f"{label}: missing or duplicate headers")
        rows = list(reader)
    except UnicodeDecodeError as exc:
        raise AuditError(f"{label}: input is not UTF-8") from exc
    if any(None in row for row in rows):
        raise AuditError(f"{label}: row has more fields than the header")
    return rows


def median(values: list[float]) -> float | None:
    vals = sorted(values)
    if not vals:
        return None
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def number(row: dict[str, str], field: str, label: str) -> float:
    try:
        x = float(row[field])
    except (KeyError, ValueError, TypeError) as exc:
        raise AuditError(f"{label}: invalid numeric field {field}") from exc
    if not math.isfinite(x) or x < 0:
        raise AuditError(f"{label}: {field} must be finite and non-negative")
    return x


def index_manifest(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    required = {"problem_id", "book", "target_count", "target_declarations"}
    if not rows or not required.issubset(rows[0]):
        raise AuditError("manifest: required columns missing")
    result = {}
    for row in rows:
        key, book = row["problem_id"].strip(), row["book"].strip()
        if not key or not book or key in result:
            raise AuditError(f"manifest: empty or duplicate task ID {key!r}")
        try:
            count = int(row["target_count"])
        except ValueError as exc:
            raise AuditError(f"manifest: invalid target_count for {key}") from exc
        declarations = [x for x in row["target_declarations"].split(";") if x]
        if count < 1 or count != len(declarations) or len(declarations) != len(set(declarations)):
            raise AuditError(f"manifest: target declaration count mismatch for {key}")
        result[key] = row
    if len(result) != 341 or len({r["book"] for r in result.values()}) != 24:
        raise AuditError("manifest must contain exactly 341 tasks across 24 source groups")
    if sum(int(r["target_count"]) for r in result.values()) != 1155:
        raise AuditError("manifest graded-declaration total must equal 1,155")
    return result


def index_model(rows: list[dict[str, str]], manifest: dict[str, dict[str, str]], label: str):
    required = {"problem_id", "textbook", "passed", "end_reason", *NUMERIC}
    if not rows or not required.issubset(rows[0]):
        raise AuditError(f"{label}: required columns missing")
    result = {}
    for row in rows:
        key = row["problem_id"].strip()
        if not key or key in result or key not in manifest:
            raise AuditError(f"{label}: empty, duplicate, or unknown task ID {key!r}")
        if row["textbook"] != manifest[key]["book"] or row["passed"] not in {"true", "false"}:
            raise AuditError(f"{label}: source group or outcome mismatch for {key}")
        for field in NUMERIC:
            number(row, field, f"{label}/{key}")
        if not row["end_reason"].strip():
            raise AuditError(f"{label}: missing end_reason for {key}")
        result[key] = row
    if set(result) != set(manifest):
        raise AuditError(f"{label}: task set differs from the pinned manifest")
    expected = SOLVE_COUNTS[label]
    actual = sum(r["passed"] == "true" for r in result.values())
    if actual != expected:
        raise AuditError(f"{label}: expected {expected} solved tasks, got {actual}")
    return result


def summarize(rows: dict[str, dict[str, str]]) -> dict[str, object]:
    all_rows = list(rows.values())
    solved = [r for r in all_rows if r["passed"] == "true"]
    failed = [r for r in all_rows if r["passed"] == "false"]
    cost = lambda subset: sum(float(r["cost_usd"]) for r in subset)
    total, good, bad = cost(all_rows), cost(solved), cost(failed)
    mean = lambda subset, field: sum(float(r[field]) for r in subset) / len(subset) if subset else None
    med = lambda subset, field: median([float(r[field]) for r in subset])
    return {
        "tasks": len(all_rows), "solved": len(solved), "unresolved": len(failed),
        "solve_rate": len(solved) / len(all_rows), "recorded_cost_usd": total,
        "recorded_cost_per_solved_task_usd": total / len(solved) if solved else None,
        "unresolved_cost_usd": bad, "unresolved_cost_share": bad / total if total else 0,
        "solved_mean_cost_usd": mean(solved, "cost_usd"),
        "unresolved_mean_cost_usd": mean(failed, "cost_usd"),
        "unresolved_to_solved_mean_cost_ratio": (mean(failed, "cost_usd") / mean(solved, "cost_usd")) if failed and solved and good else None,
        "solved_median_cost_usd": med(solved, "cost_usd"),
        "unresolved_median_cost_usd": med(failed, "cost_usd"),
        "solved_mean_input_tokens": mean(solved, "input_tokens"),
        "unresolved_mean_input_tokens": mean(failed, "input_tokens"),
        "solved_median_input_tokens": med(solved, "input_tokens"),
        "unresolved_median_input_tokens": med(failed, "input_tokens"),
        "solved_mean_turns": mean(solved, "turns"),
        "unresolved_mean_turns": mean(failed, "turns"),
        "solved_median_turns": med(solved, "turns"),
        "unresolved_median_turns": med(failed, "turns"),
        "solved_mean_added_code_loc": mean(solved, "added_code_loc"),
        "unresolved_mean_added_code_loc": mean(failed, "added_code_loc"),
        "unresolved_end_reasons": dict(sorted(Counter(r["end_reason"] for r in failed).items())),
    }


def residual_by_group(rows: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    groups = defaultdict(list)
    for row in rows.values():
        groups[row["textbook"]].append(row)
    output = []
    for name, group in groups.items():
        solved = sum(r["passed"] == "true" for r in group)
        failed = [r for r in group if r["passed"] == "false"]
        output.append({
            "source_group": name, "tasks": len(group), "solved": solved,
            "unresolved": len(failed), "solve_rate": solved / len(group),
            "unresolved_cost_usd": sum(float(r["cost_usd"]) for r in failed),
        })
    return sorted(output, key=lambda x: (-int(x["unresolved"]), str(x["source_group"])))


def target_complexity_analysis(
    manifest: dict[str, dict[str, str]],
    primary: dict[str, dict[str, str]],
    bootstrap_replicates: int = 50_000,
) -> dict[str, object]:
    """Describe task-size strata and a source-group-clustered high/low contrast.

    target_count is the count of graded declarations, not a direct count of all
    required helper lemmas. It is used only as a workload proxy.
    """
    if bootstrap_replicates < 200:
        raise AuditError("complexity bootstrap needs at least 200 replicates")

    buckets = [
        ("one_declaration", lambda n: n == 1),
        ("two_to_four_declarations", lambda n: 2 <= n <= 4),
        ("five_or_more_declarations", lambda n: n >= 5),
    ]
    bucket_reports = []
    for label, predicate in buckets:
        subset = [
            (manifest[task_id], row)
            for task_id, row in primary.items()
            if predicate(int(manifest[task_id]["target_count"]))
        ]
        if not subset:
            raise AuditError(f"no tasks in target-count stratum {label}")
        solved = sum(row["passed"] == "true" for _, row in subset)
        costs = [number(row, "cost_usd", label) for _, row in subset]
        tokens = [number(row, "input_tokens", label) for _, row in subset]
        turns = [number(row, "turns", label) for _, row in subset]
        code_loc = [number(row, "added_code_loc", label) for _, row in subset]
        bucket_reports.append({
            "stratum": label,
            "tasks": len(subset),
            "solved": solved,
            "unresolved": len(subset) - solved,
            "solve_rate": solved / len(subset),
            "mean_recorded_cost_usd": sum(costs) / len(costs),
            "median_recorded_cost_usd": median(costs),
            "mean_input_tokens": sum(tokens) / len(tokens),
            "mean_turns": sum(turns) / len(turns),
            "mean_added_code_loc": sum(code_loc) / len(code_loc),
        })

    per_group: dict[str, dict[str, int]] = defaultdict(
        lambda: {"five_plus_tasks": 0, "five_plus_unresolved": 0,
                 "under_five_tasks": 0, "under_five_unresolved": 0}
    )
    for task_id, manifest_row in manifest.items():
        if task_id not in primary:
            raise AuditError(f"primary result missing task {task_id}")
        high = int(manifest_row["target_count"]) >= 5
        failed = primary[task_id]["passed"] == "false"
        stats = per_group[manifest_row["book"]]
        if high:
            stats["five_plus_tasks"] += 1
            stats["five_plus_unresolved"] += int(failed)
        else:
            stats["under_five_tasks"] += 1
            stats["under_five_unresolved"] += int(failed)

    overlap = [
        row for row in per_group.values()
        if row["five_plus_tasks"] > 0 and row["under_five_tasks"] > 0
    ]
    if len(overlap) < 2:
        raise AuditError("fewer than two source groups contain both target-count strata")

    def pooled(groups: list[dict[str, int]]) -> tuple[int, int, int, int]:
        high_n = sum(x["five_plus_tasks"] for x in groups)
        high_f = sum(x["five_plus_unresolved"] for x in groups)
        low_n = sum(x["under_five_tasks"] for x in groups)
        low_f = sum(x["under_five_unresolved"] for x in groups)
        return high_n, high_f, low_n, low_f

    high_n, high_f, low_n, low_f = pooled(overlap)
    high_rate, low_rate = high_f / high_n, low_f / low_n
    observed_difference = high_rate - low_rate
    observed_ratio = high_rate / low_rate if low_rate > 0 else None

    # Deterministic xorshift32 cluster bootstrap. Resample source groups, not
    # individual tasks, and retain only groups with both strata for this
    # within-source comparison.
    state = 0x8BADF00D

    def next_index(limit: int) -> int:
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        state ^= state >> 17
        state &= 0xFFFFFFFF
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        return min(limit - 1, int((state / 4294967296.0) * limit))

    differences: list[float] = []
    ratios: list[float] = []
    for _ in range(bootstrap_replicates):
        sample = [overlap[next_index(len(overlap))] for _ in range(len(overlap))]
        b_high_n, b_high_f, b_low_n, b_low_f = pooled(sample)
        b_high_rate, b_low_rate = b_high_f / b_high_n, b_low_f / b_low_n
        differences.append(b_high_rate - b_low_rate)
        if b_low_rate > 0:
            ratios.append(b_high_rate / b_low_rate)

    def percentile(values: list[float], q: float) -> float:
        if not values:
            raise AuditError("bootstrap produced no valid replicates for a required interval")
        ordered = sorted(values)
        return ordered[int((len(ordered) - 1) * q)]

    return {
        "workload_proxy": "manifest graded-declaration count; not a complete measure of formalization complexity",
        "task_strata": bucket_reports,
        "five_plus_vs_under_five_within_source_groups": {
            "source_groups_with_both_strata": len(overlap),
            "high": {"tasks": high_n, "unresolved": high_f, "failure_rate": high_rate},
            "under_five": {"tasks": low_n, "unresolved": low_f, "failure_rate": low_rate},
            "risk_difference_high_minus_low": observed_difference,
            "risk_ratio_high_over_low": observed_ratio,
            "cluster_bootstrap": {
                "resampling_unit": "source_group",
                "groups_per_replicate": len(overlap),
                "replicates": bootstrap_replicates,
                "seed_hex": "8badf00d",
                "risk_difference_ci95": [
                    percentile(differences, 0.025), percentile(differences, 0.975)
                ],
                "risk_ratio_ci95": [
                    percentile(ratios, 0.025), percentile(ratios, 0.975)
                ],
                "valid_risk_ratio_replicates": len(ratios),
            },
        },
    }


def paired(reference, other):
    ref_only = sum(reference[k]["passed"] == "true" and other[k]["passed"] == "false" for k in reference)
    other_only = sum(reference[k]["passed"] == "false" and other[k]["passed"] == "true" for k in reference)
    return {"reference_only_solved": ref_only, "other_only_solved": other_only,
            "same_outcome": len(reference) - ref_only - other_only,
            "discordant_tasks": ref_only + other_only}


def audit(repo: Path) -> dict[str, object]:
    if not (repo / ".git").exists():
        raise AuditError(f"not a Git repository: {repo}")
    try:
        subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{PIN}^{{commit}}"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise AuditError("pinned release commit is unavailable; fetch it first") from exc
    raw_files, observed = {}, {}
    for path, expected in FILES.items():
        raw = read_git_file(repo, path)
        digest = git_blob_sha1(raw)
        if digest != expected:
            raise AuditError(f"Git blob mismatch for {path}: {digest}")
        raw_files[path], observed[path] = raw, digest
    manifest = index_manifest(parse_csv(raw_files["MANIFEST.csv"], "MANIFEST.csv"))
    models = {}
    for path in list(FILES)[1:]:
        name = Path(path).stem
        models[name] = index_model(parse_csv(raw_files[path], path), manifest, name)
    primary = models["gpt-6-astra"]
    residual = residual_by_group(primary)
    complexity_by_model = {
        name: target_complexity_analysis(manifest, rows)
        for name, rows in sorted(models.items())
    }
    n_failed = sum(int(x["unresolved"]) for x in residual)
    top6 = sum(int(x["unresolved"]) for x in residual[:6])
    return {
        "schema_version": "pitb-autoformbench-public-audit/0.1",
        "source": {"repository": "facebookresearch/autoformbench", "commit": PIN,
                   "scope": "manifest and public task-level leaderboard CSVs only",
                   "git_blob_sha1": observed},
        "manifest": {"tasks": len(manifest), "source_groups": len({r["book"] for r in manifest.values()}),
                     "graded_declarations": sum(int(r["target_count"]) for r in manifest.values())},
        "models": {k: summarize(v) for k, v in sorted(models.items())},
        "paired_descriptive_outcomes_vs_gpt_6_astra": {
            k: paired(primary, v) for k, v in models.items() if k != "gpt-6-astra"
        },
        "gpt_6_astra_residuals_by_source_group": residual,
        "target_count_complexity_by_model": complexity_by_model,
        "residual_concentration": {"unresolved_tasks": n_failed, "top_six_groups_unresolved": top6,
                                   "top_six_share": top6 / n_failed if n_failed else 0},
        "warning": "Cross-model outcome and cost differences are descriptive, not causal or matched-budget comparisons.",
    }


def self_test():
    sample = [
        {"id": "a", "passed": "true", "cost_usd": "10", "input_tokens": "100", "turns": "2", "added_code_loc": "5", "end_reason": "success"},
        {"id": "b", "passed": "false", "cost_usd": "40", "input_tokens": "400", "turns": "8", "added_code_loc": "12", "end_reason": "budget_consumed"},
        {"id": "c", "passed": "false", "cost_usd": "20", "input_tokens": "200", "turns": "4", "added_code_loc": "8", "end_reason": "model_gave_up"},
    ]
    for row in sample:
        row["textbook"] = "one" if row["id"] != "c" else "two"
    assert git_blob_sha1(b"abc") == "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f"
    assert median([1.0, 3.0]) == 2.0 and median([1.0, 2.0, 3.0]) == 2.0
    toy_manifest = {
        "a-low": {"book": "alpha", "target_count": "1"},
        "a-high": {"book": "alpha", "target_count": "5"},
        "b-low": {"book": "beta", "target_count": "2"},
        "b-high": {"book": "beta", "target_count": "7"},
    }
    toy_primary = {
        "a-low": {"passed": "true", "cost_usd": "1", "input_tokens": "10", "turns": "1", "added_code_loc": "2"},
        "a-high": {"passed": "false", "cost_usd": "1", "input_tokens": "10", "turns": "1", "added_code_loc": "2"},
        "b-low": {"passed": "false", "cost_usd": "1", "input_tokens": "10", "turns": "1", "added_code_loc": "2"},
        "b-high": {"passed": "false", "cost_usd": "1", "input_tokens": "10", "turns": "1", "added_code_loc": "2"},
    }
    complexity = target_complexity_analysis(toy_manifest, toy_primary, bootstrap_replicates=500)
    contrast = complexity["five_plus_vs_under_five_within_source_groups"]
    assert contrast["source_groups_with_both_strata"] == 2
    assert contrast["risk_difference_high_minus_low"] == 0.5
    assert contrast["risk_ratio_high_over_low"] == 2.0
    result = summarize({r["id"]: r for r in sample})
    assert result["solved"] == 1 and result["unresolved"] == 2
    assert math.isclose(result["unresolved_cost_share"], 60 / 70)
    assert math.isclose(result["unresolved_to_solved_mean_cost_ratio"], 3)
    groups = residual_by_group({r["id"]: r for r in sample})
    assert groups[0]["source_group"] == "one" and groups[0]["unresolved"] == 1
    assert paired({"a": sample[0], "b": sample[1]}, {"a": sample[1], "b": sample[0]}) == {
        "reference_only_solved": 1, "other_only_solved": 1, "same_outcome": 0, "discordant_tasks": 2}
    try:
        number({"x": "NaN"}, "x", "self-test")
    except AuditError:
        pass
    else:
        raise AssertionError("non-finite numeric value accepted")
    print("SELF_TEST_OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, help="local AutoformBench Git clone with the pinned commit fetched")
    parser.add_argument("--self-test", action="store_true", help="run synthetic tests without benchmark files")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            if args.repo is None:
                return 0
        if args.repo is None:
            parser.error("--repo is required unless --self-test is used alone")
        print(json.dumps(audit(args.repo.resolve()), indent=2, sort_keys=True))
        return 0
    except AuditError as exc:
        print(f"AUDIT_INVALID: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
