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
    header = f"blob {len(raw)}\\0".encode("ascii")
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
        digest = hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()
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
    assert git_blob_sha1(b"abc") == "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f"\n    assert median([1.0, 3.0]) == 2.0 and median([1.0, 2.0, 3.0]) == 2.0\n    result = summarize({r["id"]: r for r in sample})
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
