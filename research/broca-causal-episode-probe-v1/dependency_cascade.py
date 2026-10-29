"""Reference truth-maintenance evaluator for dependency-aware agent memory.

Research-only evaluator, not production Broca integration. A claim is current
only when it is the latest revision of that claim and every dependency points
to the latest, itself-current revision. Stale derived values remain auditable
but are excluded from the current view.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "fixtures" / "mutations" / "dependency_snapshots.jsonl"
MAX_LINE_BYTES = 32 * 1024 * 1024
REPORT_SCHEMA = "broca-dependency-cascade-report-v1"
VALID_STATES = {"current", "historical", "stale_dependency", "unknown_dependency"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_jsonl_bytes(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    payload = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(payload.splitlines(), 1):
        if len(raw) > MAX_LINE_BYTES:
            raise ValueError(f"{path}:{line_number}: line exceeds byte bound")
        if not raw.strip():
            raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: record must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: input must not be empty")
    return rows, payload


def _claim_key(value: dict[str, Any], context: str) -> tuple[str, int]:
    claim_id = value.get("claim_id")
    revision = value.get("revision")
    if not isinstance(claim_id, str) or not claim_id.strip():
        raise ValueError(f"{context}: claim_id must be a non-empty string")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError(f"{context}: revision must be a positive integer")
    return claim_id, revision


def _validate_and_index(
    snapshot: dict[str, Any],
) -> tuple[str, dict[tuple[str, int], dict[str, Any]], dict[str, int]]:
    snapshot_id = snapshot.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ValueError("snapshot_id must be a non-empty string")
    claims = snapshot.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError(f"{snapshot_id}: claims must be a non-empty array")

    versions: dict[tuple[str, int], dict[str, Any]] = {}
    latest: dict[str, int] = {}
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            raise ValueError(f"{snapshot_id}: claim {index} must be an object")
        key = _claim_key(claim, f"{snapshot_id} claim {index}")
        if key in versions:
            raise ValueError(f"{snapshot_id}: duplicate claim revision {key}")
        if not isinstance(claim.get("value"), str):
            raise ValueError(f"{snapshot_id} {key}: value must be a string")
        dependencies = claim.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise ValueError(f"{snapshot_id} {key}: dependencies must be an array")
        parsed_dependencies: list[tuple[str, int]] = []
        for dep_index, dependency in enumerate(dependencies):
            if not isinstance(dependency, dict):
                raise ValueError(f"{snapshot_id} {key}: dependency {dep_index} must be an object")
            dep_key = _claim_key(dependency, f"{snapshot_id} {key} dependency {dep_index}")
            parsed_dependencies.append(dep_key)
        if len(parsed_dependencies) != len(set(parsed_dependencies)):
            raise ValueError(f"{snapshot_id} {key}: duplicate dependency reference")
        versions[key] = {**claim, "_dependency_keys": parsed_dependencies}
        latest[key[0]] = max(latest.get(key[0], 0), key[1])

    return snapshot_id, versions, latest


def _check_acyclic(
    snapshot_id: str,
    versions: dict[tuple[str, int], dict[str, Any]],
) -> None:
    colors: dict[tuple[str, int], int] = {}

    def visit(key: tuple[str, int]) -> None:
        color = colors.get(key, 0)
        if color == 1:
            raise ValueError(f"{snapshot_id}: dependency cycle contains {key}")
        if color == 2:
            return
        colors[key] = 1
        claim = versions[key]
        for dependency_key in claim["_dependency_keys"]:
            if dependency_key in versions:
                visit(dependency_key)
        colors[key] = 2

    for key in sorted(versions):
        visit(key)


def evaluate_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Classify every claim revision and provide an explicit current-state view.

    Missing dependencies fail closed as unknown_dependency. A latest revision
    that depends on an older revision, or on a non-current dependency, becomes
    stale_dependency. Historical revisions remain auditable but are never placed
    in current_claims.
    """
    snapshot_id, versions, latest = _validate_and_index(snapshot)
    _check_acyclic(snapshot_id, versions)
    memo: dict[tuple[str, int], tuple[str, list[str]]] = {}

    def status_for(key: tuple[str, int]) -> tuple[str, list[str]]:
        if key in memo:
            return memo[key]
        claim = versions[key]
        if key[1] < latest[key[0]]:
            result = ("historical", [f"revision is older than latest {latest[key[0]]}"])
            memo[key] = result
            return result

        reasons: list[str] = []
        has_unknown = False
        for dependency_key in claim["_dependency_keys"]:
            dep_claim_id, dep_revision = dependency_key
            dep_latest = latest.get(dep_claim_id)
            if dependency_key not in versions or dep_latest is None:
                has_unknown = True
                reasons.append(f"dependency {dep_claim_id}@{dep_revision} is missing")
                continue
            if dep_revision != dep_latest:
                reasons.append(
                    f"dependency {dep_claim_id}@{dep_revision} is not latest {dep_latest}"
                )
                continue
            dependency_status, dependency_reasons = status_for(dependency_key)
            if dependency_status != "current":
                reasons.append(
                    f"dependency {dep_claim_id}@{dep_revision} is {dependency_status}"
                )
                reasons.extend(
                    f"dependency {dep_claim_id}@{dep_revision}: {reason}"
                    for reason in dependency_reasons
                )

        if has_unknown:
            result = ("unknown_dependency", reasons)
        elif reasons:
            result = ("stale_dependency", reasons)
        else:
            result = ("current", [])
        memo[key] = result
        return result

    evaluations: list[dict[str, Any]] = []
    current_claims: list[dict[str, Any]] = []
    state_counts = {state: 0 for state in sorted(VALID_STATES)}
    for key in sorted(versions):
        claim = versions[key]
        state, reasons = status_for(key)
        state_counts[state] += 1
        item = {
            "claim_id": key[0],
            "revision": key[1],
            "value": claim["value"],
            "dependencies": [
                {"claim_id": dep_id, "revision": dep_revision}
                for dep_id, dep_revision in claim["_dependency_keys"]
            ],
            "status": state,
            "reasons": reasons,
            "is_latest_revision": key[1] == latest[key[0]],
        }
        evaluations.append(item)
        if item["is_latest_revision"] and state == "current":
            current_claims.append(item)

    return {
        "snapshot_id": snapshot_id,
        "claim_version_count": len(versions),
        "latest_claim_count": len(latest),
        "state_counts": state_counts,
        "evaluations": evaluations,
        "current_claims": current_claims,
        "absence_claims": [
            {
                "claim_id": claim_id,
                "latest_revision": revision,
                "status": next(
                    item["status"] for item in evaluations
                    if item["claim_id"] == claim_id and item["revision"] == revision
                ),
            }
            for claim_id, revision in sorted(latest.items())
            if next(
                item["status"] for item in evaluations
                if item["claim_id"] == claim_id and item["revision"] == revision
            ) != "current"
        ],
        "claim_ceiling": (
            "synthetic dependency-consistency reference evaluator; not a production truth "
            "maintenance engine, not causal proof, and not an end-to-end answer score"
        ),
    }


def evaluate_dataset(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    if not snapshots:
        raise ValueError("snapshots must not be empty")
    ids = [snapshot.get("snapshot_id") for snapshot in snapshots]
    if any(not isinstance(snapshot_id, str) or not snapshot_id.strip() for snapshot_id in ids):
        raise ValueError("all snapshot IDs must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("snapshot IDs must be unique")
    reports = [evaluate_snapshot(snapshot) for snapshot in snapshots]
    summary = []
    for report in reports:
        latest = [
            item for item in report["evaluations"] if item["is_latest_revision"]
        ]
        summary.append({
            "snapshot_id": report["snapshot_id"],
            "current_latest_claims": sum(item["status"] == "current" for item in latest),
            "stale_latest_claims": sum(item["status"] == "stale_dependency" for item in latest),
            "unknown_latest_claims": sum(item["status"] == "unknown_dependency" for item in latest),
            "latest_claim_count": len(latest),
        })
    return {
        "schema": REPORT_SCHEMA,
        "snapshot_count": len(reports),
        "snapshots": reports,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else None
    if output_path is not None and output_path == input_path:
        raise SystemExit("output path must differ from input")
    snapshots, input_bytes = read_jsonl_bytes(input_path)
    report = evaluate_dataset(snapshots)
    report["input_sha256"] = sha256_bytes(input_bytes)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError, UnicodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
