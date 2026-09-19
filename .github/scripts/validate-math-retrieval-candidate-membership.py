#!/usr/bin/env python3
"""Prove a qualified retrieval trace returned only frozen candidate-set members."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

VERSION = "math-retrieval-candidate-membership-report-v1"
AUTHORITY = "MeasurementOnly"


class ValidationError(ValueError):
    pass


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def load_sibling(filename: str, module_name: str):
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValidationError(f"cannot load validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ranked_ids(trace: dict) -> list[tuple[str, str]]:
    retrieval = trace["retrieval"]
    rows: list[tuple[str, str]] = []
    if retrieval["mode"] == "SingleIndex":
        for source in retrieval["single"]["ranked_source_object_digests"]:
            rows.append(("SingleIndex", source))
    elif retrieval["mode"] == "Fusion":
        for channel in retrieval["fusion"]["channels"]:
            for source in channel["ranked_source_object_digests"]:
                rows.append((channel["channel"], source))
        for source in retrieval["fusion"]["fused_ranked_source_object_digests"]:
            rows.append(("Fused", source))
    else:
        raise ValidationError("trace retrieval mode unsupported")
    for source in trace["packing"]["input_ranked_source_object_digests"]:
        rows.append(("PackingInput", source))
    return rows


def validate_membership(
    candidate_set_path: Path,
    trace_path: Path,
    bundle_path: Path,
    repo_root: Path,
) -> dict:
    candidate_mod = load_sibling(
        "validate-math-retrieval-candidate-set.py", "sym_candidate_set_validator"
    )
    trace_mod = load_sibling(
        "validate-math-retrieval-trace.py", "sym_candidate_trace_validator"
    )

    candidate_raw = candidate_set_path.read_bytes()
    candidate_doc = json.loads(candidate_raw.decode("utf-8"))
    candidate_mod.validate(candidate_doc)
    candidate_sha = digest_bytes(candidate_raw)

    trace_raw = trace_path.read_bytes()
    trace = json.loads(trace_raw.decode("utf-8"))
    trace_mod.validate_trace(trace, bundle_path, repo_root)

    graph = trace["graph"]
    if candidate_sha != graph["candidate_set_sha256"]:
        raise ValidationError(
            "candidate-set file digest differs from qualified trace candidate_set_sha256"
        )
    if candidate_doc["candidate_count"] != graph["candidate_count"]:
        raise ValidationError("candidate-set count differs from qualified trace")

    graph_profile = trace_mod.load_graph_profile()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    artifacts = graph_profile.load_bundle_artifacts(bundle, repo_root, True)
    matching_indices = [
        item["doc"]
        for item in artifacts.values()
        if item["meta"]["kind"] == "RetrievalIndex"
        and item["doc"]["candidate_universe"]["candidate_set_sha256"] == candidate_sha
    ]
    if not matching_indices:
        raise ValidationError("no qualified index resolves the candidate-set digest")

    expected_meta = (
        candidate_doc["corpus_snapshot_sha256"],
        candidate_doc["knowledge_boundary_sha256"],
        candidate_doc["candidate_eligibility_policy_sha256"],
        candidate_doc["source_identity_kind"],
        candidate_doc["canonical_order"],
        candidate_doc["candidate_count"],
    )
    for index in matching_indices:
        universe = index["candidate_universe"]
        found = (
            universe["corpus_snapshot_sha256"],
            universe["knowledge_boundary_sha256"],
            universe["candidate_eligibility_policy_sha256"],
            universe["source_identity_kind"],
            universe["canonical_candidate_order"],
            universe["candidate_count"],
        )
        if found != expected_meta:
            raise ValidationError(
                f"index {index['index_id']}: candidate-set metadata differs from materialized universe"
            )

    allowed = set(candidate_doc["candidates"])
    checked = ranked_ids(trace)
    missing = [(where, source) for where, source in checked if source not in allowed]
    if missing:
        where, source = missing[0]
        raise ValidationError(
            f"{where}: returned source is outside frozen candidate universe: {source}"
        )

    return {
        "version": VERSION,
        "authority": AUTHORITY,
        "candidate_set_sha256": candidate_sha,
        "candidate_count": candidate_doc["candidate_count"],
        "trace_sha256": digest_bytes(trace_raw),
        "checked_rank_occurrences": len(checked),
        "unique_returned_sources": len({source for _, source in checked}),
        "all_returned_sources_are_members": True,
        "all_checks_passed": True,
    }


def self_test() -> None:
    candidate_mod = load_sibling(
        "validate-math-retrieval-candidate-set.py", "sym_candidate_set_selftest"
    )
    candidate_mod.self_test()
    allowed = set(candidate_mod.fixture()["candidates"])
    d = lambda n: f"sha256:{n:064x}"
    trace = {
        "retrieval": {
            "mode": "SingleIndex",
            "single": {"ranked_source_object_digests": [d(1), d(2)]},
        },
        "packing": {"input_ranked_source_object_digests": [d(1), d(2)]},
    }
    if any(source not in allowed for _, source in ranked_ids(trace)):
        raise AssertionError("valid membership fixture rejected")
    trace["retrieval"]["single"]["ranked_source_object_digests"].append(d(99))
    if not any(source not in allowed for _, source in ranked_ids(trace)):
        raise AssertionError("out-of-universe attack was not detected")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_set", nargs="?", type=Path)
    parser.add_argument("trace", nargs="?", type=Path)
    parser.add_argument("graph_bundle", nargs="?", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("math-retrieval candidate membership pure self-test: PASS")
        return 0
    if args.candidate_set is None or args.trace is None or args.graph_bundle is None:
        parser.error("candidate_set, trace, and graph_bundle required unless --self-test")

    try:
        report = validate_membership(
            args.candidate_set, args.trace, args.graph_bundle, args.repo_root
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1

    out = json.dumps(report, sort_keys=True, separators=(",", ":"))
    if args.report:
        args.report.write_text(out + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
